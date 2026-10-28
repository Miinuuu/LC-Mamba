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


    def forward(self, input, nhwc=False):
        # nhwc=True 이면 입출력이 (B,H,W,C): 연속된 블록 사이의 NCHW <-> NHWC 복사를 생략
        shortcut = input if nhwc else input.permute(0, 2, 3, 1).contiguous()

        # 윈도우 패딩 / shift 는 SW_HSS3D 내부에서 처리
        x = self.self_attention(self.ln_1(shortcut)) # B,H,W,C

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)

        return x if nhwc else x.permute(0, 3, 1, 2).contiguous()
