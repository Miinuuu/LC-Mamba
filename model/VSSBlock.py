
import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from typing import Optional, Callable
from functools import partial
import torch.nn.functional as F
from .zorder import *
from .mamba_layer import * 
from .SS2D import * 

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(input)
        x = input * self.skip_scale + self.self_attention(x)
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()

class VSSBlock_local(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            window_size=[8,8],
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        print('VSSBlock_local*')
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.window_size= window_size

    def forward(self, input):
        shortcut=input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input=shortcut
        x_pad, _ = pad_if_needed(input, input.size(), self.window_size)# # b,h,w,c
        _, Hw, Ww, C = x_pad.shape
        x_win = window_partition(x_pad, self.window_size) # B*NW,N,C

        #input = input.permute(0, 2, 3, 1).contiguous() #b,c,h,w -> b,h,w,c
        x_win = self.ln_1(x_win) # B,N,C
        nwB = x_win.shape[0]

        x = self.self_attention(x_win.view(nwB,self.window_size[0],self.window_size[1],-1))

        #print(x.shape)
        x=x.view(nwB,self.window_size[0]*self.window_size[1],-1)
        x_back_win = window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C
        x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)
        #.view(B, H * W, -1)

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()

class VSSBlock_zorder(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D_zorder(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(input)
        x = input * self.skip_scale + self.self_attention(x)
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()






class VSSBlock_zorder_local_shift(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            mlp_ratio: float = 2.,
            window_size=[8,8],
            **kwargs,
    ):
        super().__init__()
        print('VSSBlock_zorder_local_shift')

        self.window_size=window_size
        self.shift_size=shift_size

        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)

        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D_zorder_local_shift(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate,shift_size=shift_size,window_size=window_size, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        shortcut=input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input=shortcut
        x_pad, mask = pad_if_needed2(input, input.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
        _, Hw, Ww, C = x_pad.shape

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
            shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
            #print(shift_mask.shape)
            shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
            if mask is not None:
                    mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
                    shift_mask = shift_mask*mask
            shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L
            #delata==0 -> ht-1 

            #print(shift_mask.shape)
            #shift_mask = shift_mask.masked_fill(shift_mask == 0, float(-100.0)).masked_fill(shift_mask != 0, float(0.0))

        
        else:
            if mask is not None:
                shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
            else : 
                shift_mask= None
                #shift_mask= torch.ones(B,Hw,Ww,1) #None
                #shift_mask= window_partition(shift_mask, self.window_size).permute(0,2,1)


        x_pad = self.ln_1(x_pad) # B,N,C
        x = self.self_attention(x_pad,shift_mask)

        x_back_win=x
        if self.shift_size[0] or self.shift_size[1]:
            x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()

###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################

class VSSBlock_hilbert_local_shift2(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            mlp_ratio: float = 2.,
            window_size=[8,8],
            **kwargs,
    ):
        super().__init__()
        print('VSSBlock_hilbert_local_shift2')
        print('window_size',window_size)

        self.window_size=window_size
        self.shift_size=shift_size

        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)

        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D_hilbert_shift2(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate,shift_size=shift_size, window_size=window_size , **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        shortcut=input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input=shortcut
        x_pad, mask = pad_if_needed2(input, input.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
        _, Hw, Ww, C = x_pad.shape

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
            shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
            #print(shift_mask.shape)
            shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
            if mask is not None:
                    mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
                    shift_mask = shift_mask*mask
            shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L
            #delata==0 -> ht-1 

            #print(shift_mask.shape)
            #shift_mask = shift_mask.masked_fill(shift_mask == 0, float(-100.0)).masked_fill(shift_mask != 0, float(0.0))

        
        else:
            if mask is not None:
                shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
            else : 
                shift_mask= None
                #shift_mask= torch.ones(B,Hw,Ww,1) #None
                #shift_mask= window_partition(shift_mask, self.window_size).permute(0,2,1)

        #x_win = window_partition(x_pad, self.window_size) # B*NW,N,C

        #input = input.permute(0, 2, 3, 1).contiguous() #b,c,h,w -> b,h,w,c
        x_pad = self.ln_1(x_pad) # B,N,C
        #nwB = x_win.shape[0]
        #x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1)
        x_back_win = self.self_attention(x_pad,shift_mask)

        #print(x.shape)
        #x=x.view(nwB,self.window_size[0]*self.window_size[1],-1)
        #x_back_win = window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C
          
        if self.shift_size[0] or self.shift_size[1]:
            x = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)
        #.view(B, H * W, -1)

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()
###########################################################################################################################################################################
###########################################################################################################################################################################

class VSSBlock_hilbert_local_shift(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            mlp_ratio: float = 2.,
            window_size=[8,8],
            **kwargs,
    ):
        super().__init__()
        print('VSSBlock_hilbert_local_shift')

        self.window_size=window_size
        self.shift_size=shift_size

        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)

        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D_hilbert_shift(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate,shift_size=shift_size, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        shortcut=input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input=shortcut
        x_pad, mask = pad_if_needed2(input, input.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
        _, Hw, Ww, C = x_pad.shape

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
            shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
            #print(shift_mask.shape)
            shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
            if mask is not None:
                    mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
                    shift_mask = shift_mask*mask
            shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L
            #delata==0 -> ht-1 

            #print(shift_mask.shape)
            #shift_mask = shift_mask.masked_fill(shift_mask == 0, float(-100.0)).masked_fill(shift_mask != 0, float(0.0))

        
        else:
            if mask is not None:
                shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
            else : 
                shift_mask= None
                #shift_mask= torch.ones(B,Hw,Ww,1) #None
                #shift_mask= window_partition(shift_mask, self.window_size).permute(0,2,1)

        x_win = window_partition(x_pad, self.window_size) # B*NW,N,C

        #input = input.permute(0, 2, 3, 1).contiguous() #b,c,h,w -> b,h,w,c
        x_win = self.ln_1(x_win) # B,N,C
        nwB = x_win.shape[0]
        x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1)
        x = self.self_attention(x_win,shift_mask)

        #print(x.shape)
        x=x.view(nwB,self.window_size[0]*self.window_size[1],-1)
        x_back_win = window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C
          
        if self.shift_size[0] or self.shift_size[1]:
            x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)
        #.view(B, H * W, -1)

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()
###########################################################################################################################################################################
###########################################################################################################################################################################

class VSSBlock_hilbert_local_shift_rot(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            mlp_ratio: float = 2.,
            window_size=[8,8],
            **kwargs,
    ):
        super().__init__()
        print('VSSBlock_hilbert_local_shift_rot')

        self.window_size=window_size
        self.shift_size=shift_size

        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)

        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D_hilbert_shift_rot(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate,shift_size=shift_size,window_size=window_size, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        shortcut=input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input=shortcut
        x_pad, mask = pad_if_needed2(input, input.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
        _, Hw, Ww, C = x_pad.shape

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
            shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
            #print(shift_mask.shape)
            shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
            if mask is not None:
                    mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
                    shift_mask = shift_mask*mask
            shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L
            #delata==0 -> ht-1 

            #print(shift_mask.shape)
            #shift_mask = shift_mask.masked_fill(shift_mask == 0, float(-100.0)).masked_fill(shift_mask != 0, float(0.0))

        
        else:
            if mask is not None:
                shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
            else : 
                shift_mask= None
                #shift_mask= torch.ones(B,Hw,Ww,1) #None
                #shift_mask= window_partition(shift_mask, self.window_size).permute(0,2,1)

        #x_win = window_partition(x_pad, self.window_size) # B*NW,N,C

        # #input = input.permute(0, 2, 3, 1).contiguous() #b,c,h,w -> b,h,w,c
        #nwB = x_win.shape[0]
        #x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1)
        
        x_pad = self.ln_1(x_pad) # B,N,C
        x_back_win = self.self_attention(x_pad,shift_mask)#B,H,W,C

        #print(x.shape)
        #x=x.view(nwB,self.window_size[0]*self.window_size[1],-1)
        #x_back_win = window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C
        #x_back_win=x
        if self.shift_size[0] or self.shift_size[1]:
            x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)
        #.view(B, H * W, -1)

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()
###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################


class VSSBlock_hilbert_3d_local_shift_rot(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            mlp_ratio: float = 2.,
            window_size=[8,8],
            **kwargs,
    ):
        super().__init__()
        print('VSSBlock_hilbert_3d_local_shift_rot')

        self.window_size=window_size
        self.shift_size=shift_size

        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)

        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D_hilbert_3d_shift_rot(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate,shift_size=shift_size,window_size=self.window_size , **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        shortcut=input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input=shortcut
        x_pad, mask = pad_if_needed2(input, input.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
        _, Hw, Ww, C = x_pad.shape

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
            shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
            #print(shift_mask.shape)
            shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
            if mask is not None:
                    mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
                    shift_mask = shift_mask*mask
            shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L
            #delata==0 -> ht-1 

            #print(shift_mask.shape)
            #shift_mask = shift_mask.masked_fill(shift_mask == 0, float(-100.0)).masked_fill(shift_mask != 0, float(0.0))

        
        else:
            if mask is not None:
                shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
            else : 
                shift_mask= None
                #shift_mask= torch.ones(B,Hw,Ww,1) #None
                #shift_mask= window_partition(shift_mask, self.window_size).permute(0,2,1)

        #x_win = window_partition(x_pad, self.window_size) # B*NW,N,C

        # #input = input.permute(0, 2, 3, 1).contiguous() #b,c,h,w -> b,h,w,c
        #nwB = x_win.shape[0]
        #x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1)
        
        
        x_pad = self.ln_1(x_pad) # B,N,C
        
        x_back_win = self.self_attention(x_pad,shift_mask)

        #print(x.shape)
        # x_back_win = window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C
        
        #x=x.view(nwB,self.window_size[0]*self.window_size[1],-1)
        #x_back_win= window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C

        if self.shift_size[0] or self.shift_size[1]:
            x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)
        #.view(B, H * W, -1)

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()
###########################################################################################################################################################################
###########################################################################################################################################################################



class VSSBlock_localmamba(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            window_size=[8,8],
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        print('VSSBlock_local*')
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D_localmamba(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, window_size=window_size,**kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.window_size= window_size


    def local_scan_bchw(self, x, w=7, H=14, W=14, flip=False, column_first=False):
        """Local windowed scan in LocalMamba
        Input: 
            x: [B, C, H, W]
            H, W: original width and height before padding
            column_first: column-wise scan first (the additional direction in VMamba)
        Return: [B, C, L]
        # """
        # B, C, _, _ = x.shape
        # x = x.view(B, C, H, W)
        # Hg, Wg = math.ceil(H / w), math.ceil(W / w)
        # if H % w != 0 or W % w != 0:
        #     newH, newW = Hg * w, Wg * w
        #     x = F.pad(x, (0, newW - W, 0, newH - H))
        # if column_first:
        #     x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, -1)
        # else:
        #     x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
        # if flip:
        #     x = x.flip([-1])
        # """
        B, H, W,C = x.shape
        #x = x.view(B, C, H, W)
        Hg, Wg = math.ceil(H / w), math.ceil(W / w)
        if H % w != 0 or W % w != 0:
            newH, newW = Hg * w, Wg * w
            x = F.pad(x, (0, newW - W, 0, newH - H))
        if column_first:
            x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, -1)
        else:
            x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
        if flip:
            x = x.flip([-1])
        return x


    def local_reverse(self, x, w=7, H=14, W=14, flip=False, column_first=False):
        """Local windowed scan in LocalMamba
        Input: 
            x: [B, C, L]
            H, W: original width and height before padding
            column_first: column-wise scan first (the additional direction in VMamba)
        Return: [B, C, L]
        """
        B, C, L = x.shape
        Hg, Wg = math.ceil(H / w), math.ceil(W / w)
        if flip:
            x = x.flip([-1])
        if H % w != 0 or W % w != 0:
            if column_first:
                x = x.view(B, C, Wg, Hg, w, w).permute(0, 1, 3, 5, 2, 4).reshape(B, C, Hg * w, Wg * w)
            else:
                x = x.view(B, C, Hg, Wg, w, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hg * w, Wg * w)
            x = x[:, :, :H, :W].reshape(B, C, -1)
        else:
            if column_first:
                x = x.view(B, C, Wg, Hg, w, w).permute(0, 1, 3, 5, 2, 4).reshape(B, C, L)
            else:
                x = x.view(B, C, Hg, Wg, w, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L)
        return x



    def forward(self, input):
        shortcut=input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input=shortcut
        x_pad, _ = pad_if_needed(input, input.size(), self.window_size)# # b,h,w,c

        _,H_p,W_p,_=x_pad.shape
        #x_pad = x_pad.view(B,H_p, self.window_size[0], W_p, self.window_size[1],C).permute(0,1,3,2,4,5).view(B,-1,C) # B,H_p/w, W_p/w ,w1,w2,c
    
        x_pad = self.ln_1(x_pad) # B,N,C
        x = self.self_attention(x_pad) ##B,H,W,C
        x_back=x
        #print(x.shape)
        #x=x.view(nwB,self.window_size[0]*self.window_size[1],-1) # B,N,C
        
        
        #x_back_win = window_reverse(x, self.window_size, H_p, W_p) # B,H,W,C
        
        x = depad_if_needed(x_back, shortcut.size(), self.window_size) 
        #.view(B, H * W, -1)

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()


class VSSBlock_continuousmamba(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            window_size=[8,8],
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        print('VSSBlock_continuousmamba*')
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D_continuousmamba(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, window_size=window_size,**kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.window_size= window_size



    def forward(self, input):
        shortcut=input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input=shortcut
        x_pad, _ = pad_if_needed(input, input.size(), self.window_size)# # b,h,w,c

        _,H_p,W_p,_=x_pad.shape
    
        x_pad = self.ln_1(x_pad) # B,N,C
        x = self.self_attention(x_pad) ##B,H,W,C
        x_back=x
        
        x = depad_if_needed(x_back, shortcut.size(), self.window_size) 

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()


class VSSBlock_bidirection(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            window_size=[8,8],
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        print('VSSBlock_bidirection*')
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D_bidirection(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, window_size=window_size,**kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.window_size= window_size



    def forward(self, input):
        shortcut=input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input=shortcut
        x_pad, _ = pad_if_needed(input, input.size(), self.window_size)# # b,h,w,c

        _,H_p,W_p,_=x_pad.shape
    
        x_pad = self.ln_1(x_pad) # B,N,C
        x = self.self_attention(x_pad) ##B,H,W,C
        x_back=x
        
        x = depad_if_needed(x_back, shortcut.size(), self.window_size) 

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()




class VSSBlock_cross(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(input)
        x = input * self.skip_scale + self.self_attention(x)
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()

class VSSBlock_local(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            window_size=[8,8],
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        print('VSSBlock_local*')
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.window_size= window_size

    def forward(self, input):
        shortcut=input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input=shortcut
        x_pad, _ = pad_if_needed(input, input.size(), self.window_size)# # b,h,w,c
        _, Hw, Ww, C = x_pad.shape
        x_win = window_partition(x_pad, self.window_size) # B*NW,N,C

        #input = input.permute(0, 2, 3, 1).contiguous() #b,c,h,w -> b,h,w,c
        x_win = self.ln_1(x_win) # B,N,C
        nwB = x_win.shape[0]

        x = self.self_attention(x_win.view(nwB,self.window_size[0],self.window_size[1],-1))

        #print(x.shape)
        x=x.view(nwB,self.window_size[0]*self.window_size[1],-1)
        x_back_win = window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C
        x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)
        #.view(B, H * W, -1)

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()
    

class VSSBlock_hilbert_local_shift_rot_inv(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            shift_size=0,
            mlp_ratio: float = 2.,
            window_size=[8,8],
            **kwargs,
    ):
        super().__init__()
        print('VSSBlock_hilbert_local_shift_rot_inv')

        self.window_size=window_size
        self.shift_size=shift_size

        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)

        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D_hilbert_shift_rot_inv(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate,shift_size=shift_size, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        shortcut=input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        input=shortcut
        x_pad, mask = pad_if_needed2(input, input.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
        _, Hw, Ww, C = x_pad.shape

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
            shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
            #print(shift_mask.shape)
            shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
            if mask is not None:
                    mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
                    shift_mask = shift_mask*mask
            shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L
            #delata==0 -> ht-1 

            #print(shift_mask.shape)
            #shift_mask = shift_mask.masked_fill(shift_mask == 0, float(-100.0)).masked_fill(shift_mask != 0, float(0.0))

        
        else:
            if mask is not None:
                shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
            else : 
                shift_mask= None
                #shift_mask= torch.ones(B,Hw,Ww,1) #None
                #shift_mask= window_partition(shift_mask, self.window_size).permute(0,2,1)

        x_win = window_partition(x_pad, self.window_size) # B*NW,N,C

        #input = input.permute(0, 2, 3, 1).contiguous() #b,c,h,w -> b,h,w,c
        x_win = self.ln_1(x_win) # B,N,C
        nwB = x_win.shape[0]
        x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1)
        x = self.self_attention(x_win,shift_mask)

        #print(x.shape)
        x=x.view(nwB,self.window_size[0]*self.window_size[1],-1)
        x_back_win = window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C
          
        if self.shift_size[0] or self.shift_size[1]:
            x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)
        #.view(B, H * W, -1)

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()
    

# ###########################################################################################################################################################################
# ###########################################################################################################################################################################

# class VSSBlock_hilbert_3d_local_shift(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             shift_size=0,
#             mlp_ratio: float = 2.,
#             window_size=[8,8],
#             **kwargs,
#     ):
#         super().__init__()
#         print('VSSBlock_hilbert_3d_local_shift')

#         self.window_size=window_size
#         self.shift_size=shift_size

#         if not isinstance(self.window_size, (tuple, list)):
#             self.window_size = to_2tuple(window_size)

#         if not isinstance(self.shift_size, (tuple, list)):
#             self.shift_size = to_2tuple(shift_size)

#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = SS2D_hilbert_3d_shift(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate,shift_size=shift_size,window_size=window_size, **kwargs)
#         self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
#         self.conv_blk = CAB(hidden_dim)
#         self.ln_2 = nn.LayerNorm(hidden_dim)
#         self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


#     def forward(self, input):
#         shortcut=input.permute(0, 2, 3, 1).contiguous()
#         B,H,W,C= shortcut.shape
#         input=shortcut
#         x_pad, mask = pad_if_needed2(input, input.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
#         _, Hw, Ww, C = x_pad.shape

#         if self.shift_size[0] or self.shift_size[1]:
#             _, H_p, W_p, C = x_pad.shape
#             x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
#             shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
#             #print(shift_mask.shape)
#             shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
#             if mask is not None:
#                     mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
#                     shift_mask = shift_mask*mask
#             shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L
#             #delata==0 -> ht-1 

#             #print(shift_mask.shape)
#             #shift_mask = shift_mask.masked_fill(shift_mask == 0, float(-100.0)).masked_fill(shift_mask != 0, float(0.0))

        
#         else:
#             if mask is not None:
#                 shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
#             else : 
#                 shift_mask= None
#                 #shift_mask= torch.ones(B,Hw,Ww,1) #None
#                 #shift_mask= window_partition(shift_mask, self.window_size).permute(0,2,1)

#         x_win = window_partition(x_pad, self.window_size) # B*NW,N,C

#         #input = input.permute(0, 2, 3, 1).contiguous() #b,c,h,w -> b,h,w,c
#         x_win = self.ln_1(x_win) # B,N,C
#         nwB = x_win.shape[0]
#         x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1)
#         x = self.self_attention(x_win,shift_mask)

#         #print(x.shape)
#         x=x.view(nwB,self.window_size[0]*self.window_size[1],-1)
#         x_back_win = window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C
          
#         if self.shift_size[0] or self.shift_size[1]:
#             x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

#         x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)
#         #.view(B, H * W, -1)

#         x= shortcut * self.skip_scale + x  # B,H,W,C

#         x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

#         return x.permute(0, 3, 1, 2).contiguous()


###########################################################################################################################################################################
############################################################################################################################################################################

# class VSSBlock_v2_local_shift(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             shift_size=0,
#             mlp_ratio: float = 2.,
#             window_size=[8,8],
#             **kwargs,
#     ):
#         super().__init__()
#         print('VSSBlock_v2_local_shift')

#         self.window_size=window_size
#         self.shift_size=shift_size

#         if not isinstance(self.window_size, (tuple, list)):
#             self.window_size = to_2tuple(window_size)

#         if not isinstance(self.shift_size, (tuple, list)):
#             self.shift_size = to_2tuple(shift_size)

#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = SS2D_v2_shift(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate,shift_size=shift_size, **kwargs)
#         self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
#         self.conv_blk = CAB(hidden_dim)
#         self.ln_2 = nn.LayerNorm(hidden_dim)
#         self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


#     def forward(self, input):
#         shortcut=input.permute(0, 2, 3, 1).contiguous()
#         B,H,W,C= shortcut.shape
#         input=shortcut
#         x_pad, mask = pad_if_needed2(input, input.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
#         #x_pad, mask = pad_if_needed(input, input.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
#         _, Hw, Ww, C = x_pad.shape

#         if self.shift_size[0] or self.shift_size[1]:
#             _, H_p, W_p, C = x_pad.shape
#             x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
#             shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
#             #print(shift_mask.shape)
#             shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
#             if mask is not None:
#                     mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
#                     shift_mask = shift_mask*mask
#             shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1 ,L
#             #delata==0 -> ht-1 
#             #print(shift_mask.shape)
#             #shift_mask = shift_mask.masked_fill(shift_mask == 0, float(-100.0)).masked_fill(shift_mask != 0, float(0.0))
#         else:
#             if mask is not None:
#                 shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
#             else : 
#                 shift_mask= None
#                 #shift_mask= torch.ones(B,Hw,Ww,1) #None
#                 #shift_mask= window_partition(shift_mask, self.window_size).permute(0,2,1)
#         # if self.shift_size[0] or self.shift_size[1]:
#         #     _, H_p, W_p, C = x_pad.shape
#         #     x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
#         #     cor_pad = torch.roll(cor_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            
#         #     if hasattr(self, 'HW') and self.HW.item() == H_p * W_p: 
#         #         shift_mask = self.attn_mask
#         #     else:
#         #         shift_mask = torch.zeros((1, H_p, W_p, 1))  # 1 H W 1
#         #         h_slices = (slice(0, -self.window_size[0]),
#         #                     slice(-self.window_size[0], -self.shift_size[0]),
#         #                     slice(-self.shift_size[0], None))
#         #         w_slices = (slice(0, -self.window_size[1]),
#         #                     slice(-self.window_size[1], -self.shift_size[1]),
#         #                     slice(-self.shift_size[1], None))
#         #         cnt = 0
#         #         for h in h_slices:
#         #             for w in w_slices:
#         #                 shift_mask[:, h, w, :] = cnt
#         #                 cnt += 1

#         #         mask_windows = window_partition(shift_mask, self.window_size).squeeze(-1)  
#         #         shift_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#         #         shift_mask = shift_mask.masked_fill(shift_mask != 0, 
#         #                         float(-100.0)).masked_fill(shift_mask == 0, 
#         #                         float(0.0))
                                
#         #         if mask is not None:
#         #             shift_mask = shift_mask.masked_fill(mask != 0, 
#         #                         float(-100.0))
#         #         #self.register_buffer("attn_mask", shift_mask)
#         #         #self.register_buffer("HW", torch.Tensor([H_p*W_p]))
#         # else: 
#         #     shift_mask = mask
        
#         # if shift_mask is not None:
#         #     shift_mask = shift_mask.to(x_pad.device)
#         x_win = window_partition(x_pad, self.window_size) # B*NW,N,C

#         #input = input.permute(0, 2, 3, 1).contiguous() #b,c,h,w -> b,h,w,c
#         x_win = self.ln_1(x_win) # B,N,C
#         nwB = x_win.shape[0]
#         x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1) #bnw,w0,w1,c
#         x = self.self_attention(x_win,shift_mask) #bnw,w0,w1,c

#         #print(x.shape)
#         x=x.view(nwB,self.window_size[0]*self.window_size[1],-1)
#         x_back_win = window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C
          
#         if self.shift_size[0] or self.shift_size[1]:
#             x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

#         x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)
#         #.view(B, H * W, -1)

#         x= shortcut * self.skip_scale + x  # B,H,W,C

#         x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

#         return x.permute(0, 3, 1, 2).contiguous()

# ###########################################################################################################################################################################
# ###########################################################################################################################################################################



# class VSSBlock2_local_shift(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             shift_size=0,
#             mlp_ratio: float = 2.,
#             window_size=[8,8],
#             **kwargs,
#     ):
#         super().__init__()
#         print('VSSBlock2_local_shift')

#         self.window_size=window_size
#         self.shift_size=shift_size

#         if not isinstance(self.window_size, (tuple, list)):
#             self.window_size = to_2tuple(window_size)

#         if not isinstance(self.shift_size, (tuple, list)):
#             self.shift_size = to_2tuple(shift_size)

#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = SS2D2_shiftmask(d_model=hidden_dim, d_state=64,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
#         #self.self_attention = SS2D2(d_model=hidden_dim, d_state=64,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
#         self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
#         self.conv_blk = CAB(hidden_dim)
#         self.ln_2 = nn.LayerNorm(hidden_dim)
#         self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


#     def forward(self, input):
#         shortcut=input.permute(0, 2, 3, 1).contiguous()
#         B,H,W,C= shortcut.shape
#         input=shortcut
#         x_pad, mask = pad_if_needed2(input, input.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
#         _, Hw, Ww, C = x_pad.shape

#         if self.shift_size[0] or self.shift_size[1]:
#             _, H_p, W_p, C = x_pad.shape
#             x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
#             shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
#             #print(shift_mask.shape)
#             shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
#             if mask is not None:
#                     mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
#                     shift_mask = shift_mask*mask
#             shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L
#             #delata==0 -> ht-1 

#             #print(shift_mask.shape)
#             #shift_mask = shift_mask.masked_fill(shift_mask == 0, float(-100.0)).masked_fill(shift_mask != 0, float(0.0))

        
#         else:
#             if mask is not None:
#                 shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
#             else : 
#                 shift_mask= None
#                 #shift_mask= torch.ones(B,Hw,Ww,1) #None
#                 #shift_mask= window_partition(shift_mask, self.window_size).permute(0,2,1)

#         x_win = window_partition(x_pad, self.window_size) # B*NW,N,C

#         #input = input.permute(0, 2, 3, 1).contiguous() #b,c,h,w -> b,h,w,c
#         x_win = self.ln_1(x_win) # B,N,C
#         nwB = x_win.shape[0]
#         x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1)
#         x = self.self_attention(x_win,shift_mask)

#         #print(x.shape)
#         x=x.view(nwB,self.window_size[0]*self.window_size[1],-1)
#         x_back_win = window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C
          
#         if self.shift_size[0] or self.shift_size[1]:
#             x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

#         x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)
#         #.view(B, H * W, -1)

#         x= shortcut * self.skip_scale + x  # B,H,W,C

#         x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

#         return x.permute(0, 3, 1, 2).contiguous()

# ###########################################################################################################################################################################
# ###########################################################################################################################################################################

   
# class VSSBlock_hilbert_3d_local_shift2(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             shift_size=0,
#             mlp_ratio: float = 2.,
#             window_size=[8,8],
#             **kwargs,
#     ):
#         super().__init__()
#         print('VSSBlock_hilbert_3d_local_shift2')

#         self.window_size=window_size
#         self.shift_size=shift_size

#         if not isinstance(self.window_size, (tuple, list)):
#             self.window_size = to_2tuple(window_size)

#         if not isinstance(self.shift_size, (tuple, list)):
#             self.shift_size = to_2tuple(shift_size)

#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = SS2D_hilbert_3d_shift2(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate,shift_size=shift_size, **kwargs)
#         self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
#         self.conv_blk = CAB(hidden_dim)
#         self.ln_2 = nn.LayerNorm(hidden_dim)
#         self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


#     def forward(self, input):
#         shortcut=input.permute(0, 2, 3, 1).contiguous()
#         B,H,W,C= shortcut.shape
#         input=shortcut
#         x_pad, mask = pad_if_needed2(input, input.size(), self.window_size)# # b,hw,ww,c , b,hw,ww,1
#         _, Hw, Ww, C = x_pad.shape

#         if self.shift_size[0] or self.shift_size[1]:
#             _, H_p, W_p, C = x_pad.shape
#             x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
#             shift_mask = torch.zeros((B, H_p, W_p, 1))  # 1 H W 1
#             #print(shift_mask.shape)
#             shift_mask[:,  : int(-self.shift_size[0]) , : int(-self.shift_size[1])  , :]  = 1.# B H_p W_p 1
#             if mask is not None:
#                     mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로 
#                     shift_mask = shift_mask*mask
#             shift_mask = window_partition(shift_mask, self.window_size).permute(0,2,1)   #(b*nW),1(d),L
#             #delata==0 -> ht-1 

#             #print(shift_mask.shape)
#             #shift_mask = shift_mask.masked_fill(shift_mask == 0, float(-100.0)).masked_fill(shift_mask != 0, float(0.0))

        
#         else:
#             if mask is not None:
#                 shift_mask= window_partition(mask, self.window_size).permute(0,2,1)
#             else : 
#                 shift_mask= None
#                 #shift_mask= torch.ones(B,Hw,Ww,1) #None
#                 #shift_mask= window_partition(shift_mask, self.window_size).permute(0,2,1)

#         x_win = window_partition(x_pad, self.window_size) # B*NW,N,C

#         #input = input.permute(0, 2, 3, 1).contiguous() #b,c,h,w -> b,h,w,c
#         x_win = self.ln_1(x_win) # B,N,C
#         nwB = x_win.shape[0]
#         x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1)
#         x = self.self_attention(x_win,shift_mask)

#         #print(x.shape)
#         x=x.view(nwB,self.window_size[0]*self.window_size[1],-1)
#         x_back_win = window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C
          
#         if self.shift_size[0] or self.shift_size[1]:
#             x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

#         x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)
#         #.view(B, H * W, -1)

#         x= shortcut * self.skip_scale + x  # B,H,W,C

#         x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

#         return x.permute(0, 3, 1, 2).contiguous()

# ###########################################################################################################################################################################
# ###########################################################################################################################################################################

# class VSSBlock_v2_local(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             shift_size=0,
#             window_size=[8,8],
#             mlp_ratio: float = 2.,
#             **kwargs,
#     ):
#         super().__init__()
#         print('VSSBlock_v2_local')
#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = SS2D_v2(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
#         self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
#         self.conv_blk = CAB(hidden_dim)
#         self.ln_2 = nn.LayerNorm(hidden_dim)
#         self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
#         self.window_size= window_size

#     def forward(self, input):
#         shortcut=input.permute(0, 2, 3, 1).contiguous()
#         B,H,W,C= shortcut.shape
#         input=shortcut
#         x_pad, _ = pad_if_needed(input, input.size(), self.window_size)# # b,h,w,c
#         _, Hw, Ww, C = x_pad.shape
#         x_win = window_partition(x_pad, self.window_size) # B*NW,N,C

#         #input = input.permute(0, 2, 3, 1).contiguous() #b,c,h,w -> b,h,w,c
#         x_win = self.ln_1(x_win) # B,N,C
#         nwB = x_win.shape[0]

#         x = self.self_attention(x_win.view(nwB,self.window_size[0],self.window_size[1],-1))

#         #print(x.shape)
#         x=x.view(nwB,self.window_size[0]*self.window_size[1],-1)
#         x_back_win = window_reverse(x, self.window_size, Hw, Ww) # B,H,W,C
#         x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)
#         #.view(B, H * W, -1)

#         x= shortcut * self.skip_scale + x  # B,H,W,C

#         x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

#         return x.permute(0, 3, 1, 2).contiguous()


