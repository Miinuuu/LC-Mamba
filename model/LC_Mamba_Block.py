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

        self._scan_mask_cache = {} # (B,H,W,device) -> 스캔 순서 dt 마스크


    def scan_mask(self, B, H, W, device):
        """
        shift / padding 윈도우용 dt 마스크를 스캔 순서로 변환한 것 (B//2*nW, K, 1, L), 필요 없으면 None
        입력 크기에만 의존하므로 크기별로 한 번만 GPU 에서 만들고 캐시
        """
        key = (B, H, W, device)
        if key not in self._scan_mask_cache:
            ones = torch.ones(B, H, W, 1, device=device)
            ones_pad, mask = pad_if_needed(ones, ones.size(), self.window_size) # mask: b,hw,ww,1
            _, H_p, W_p, _ = ones_pad.shape

            if self.shift_size[0] or self.shift_size[1]:
                shift_mask = torch.zeros((B, H_p, W_p, 1), device=device)  # 1 H W 1
                shift_mask[:,  : H_p - self.shift_size[0] , : W_p - self.shift_size[1]  , :]  = 1.# B H_p W_p 1 (roll 로 넘어온 영역만 0)
                if mask is not None:
                        mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로
                        shift_mask = shift_mask*mask
                mask = shift_mask

            if mask is not None:
                mask = self.self_attention.cross_scan(mask.permute(0, 3, 1, 2)) # B//2*nW,K,1,L

            if len(self._scan_mask_cache) >= 8:
                self._scan_mask_cache.clear()
            self._scan_mask_cache[key] = mask
        return self._scan_mask_cache[key]


    def forward(self, input, nhwc=False):
        # nhwc=True 이면 입출력이 (B,H,W,C): 연속된 블록 사이의 NCHW <-> NHWC 복사를 생략
        shortcut = input if nhwc else input.permute(0, 2, 3, 1).contiguous()
        B,H,W,C= shortcut.shape
        x_pad, _ = pad_if_needed(shortcut, shortcut.size(), self.window_size, return_mask=False)# # b,hw,ww,c

        if self.shift_size[0] or self.shift_size[1]:
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) # 반시계방향으로

        shift_mask = self.scan_mask(B, H, W, shortcut.device)

        x_pad = self.ln_1(x_pad) # B,N,C
        x_back_win = self.self_attention(x_pad,shift_mask)

        if self.shift_size[0] or self.shift_size[1]:
            x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2)) # 원래 위치로 역 roll

        x = depad_if_needed(x_back_win, shortcut.size(), self.window_size)

        x= shortcut * self.skip_scale + x  # B,H,W,C

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)

        return x if nhwc else x.permute(0, 3, 1, 2).contiguous()
