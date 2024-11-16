
import torch
import torch.nn as nn
from .layers import *
from .mamba_layer import * 
from .SW_HSS3D import * 
from typing import Callable

##################################################################################################################################################

class LC_Mamba_Block(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            expand: float = 2.,
            window_size=8,
            **kwargs,
    ):
        super().__init__()
        print('LC_Mamba_Block')

        self.window_size=window_size
        self.shift_size=shift_size

        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)

        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)
        
        print('window_size',self.window_size)
        print('shift_size',self.shift_size)

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SW_HSS3D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate,shift_size=self.shift_size, window_size=self.window_size , **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        shortcut=input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input=shortcut
        x_pad, mask = pad_if_needed(input, input.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
        _, Hw, Ww, C = x_pad.shape

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
            shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
            shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
            if mask is not None:
                    mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
                    shift_mask = shift_mask*mask
            shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L
        
        else:
            if mask is not None:
                shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
            else : 
                shift_mask= None
                
        x_pad = self.ln_1(x_pad) # B,N,C
        x_back_win = self.self_attention(x_pad,shift_mask)
          
        if self.shift_size[0] or self.shift_size[1]:
            x = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()

