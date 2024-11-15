
import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from typing import Optional, Callable
from functools import partial
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref 
#from .mamba.mamba_ssm_mask.ops.selective_scan_interface import  selective_scan_mask_ref
from .zorder import *

from .hilbert_2d import * 
from .hilbert_3d import * 
from .gilbert_d2xyz import *


try:
    from .mamba2.csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

# try:
#     from .csms6s import selective_scan_fn, selective_scan_flop_jit
# except:
#     from csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from mamba2.ssd_minimal import selective_scan_chunk_fn


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn
        #self.selective_scan = selective_scan_ref

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = 2 * H * W
        K = 4
        B = B // 2
        
        #self.merge_x(x), =  B//2,C,2*H*W
        #print('forward_core')
        x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous()
        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,L,  horizon,vertical
        x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,L,
        xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,L,
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
        out_y = self.selective_scan(
            xs, 
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        '''out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)'''



        assert out_y.dtype == torch.float

        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #B//2,C,2*H*W
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # print(x.shape)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        # print(y.shape)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B//2, H*W, 2, int(self.expand*C))
        y = torch.cat([y[:, :, 0], y[:, :, 1]], 0).view(B, H, W, int(self.expand*C))#.view(B//2, 2*H, 2*W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out





class SS2D_zorder_shift(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            shift_size=0,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        print('SS2D_zorder_shift')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn
        #self.selective_scan = selective_scan_mask_ref
        #self.selective_scan = selective_scan_mask_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.delta_softplus=True
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L
    def unmerge_x(self, x ): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
       # x = x.view(B,C,2,-1) # B,,C,2,L
       # x= torch.cat( (x[:,:,0], x[:,:,1]),0).view(B*2,C,L//2).contiguous() # B,H,W,C


        odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

        # 홀수, 짝수 순서로 배치
        x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

        #print((x-merged_tensor).max())
        return x

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # BNW,C,W0,W1
        #print(shift_mask.shape if shift_mask != None else 'None')  # B//2 NW ,1 ,L
        
        z_ordered_tensor, original_shape, z_ordered_indices = z_ordering_4d(x)
        z_ordered_tensor_wh, _, z_ordered_indices_wh = z_ordering_4d(torch.transpose(x, dim0=2, dim1=3).contiguous()) 
        
        #print("Z-ordered tensor:", z_ordered_tensor)
        x=z_ordered_tensor.view(B,C,H,W).contiguous()
        x_wh=z_ordered_tensor_wh.view(B,C,W,H).contiguous() 

        L = 2 * H * W #inter 
        K = 4 #4방향 
        
        if shift_mask != None:
            shift_mask=shift_mask.view(B,1,H,W)
            #print(shift_mask.shape)
            z_ordered_mask, _, _ = z_ordering_4d(shift_mask)
            z_ordered_mask_wh, _, _ = z_ordering_4d(torch.transpose(shift_mask, dim0=2, dim1=3).contiguous()) 
            z_ordered_mask=z_ordered_mask.view(B,1,H,W).contiguous()
            z_ordered_mask_wh=z_ordered_mask_wh.view(B,1,W,H).contiguous() 
            m_hwwh = torch.stack([self.merge_x(z_ordered_mask), self.merge_x(z_ordered_mask_wh)], dim=1).view(B//2, 2, 1, L) # B//2,2,C,2L,  horizon,vertical
            m_hwwh_reverse =  torch.flip(m_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
            ms = torch.cat([m_hwwh,m_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
            shift_mask = ms.float().view(B//2, -1, L).unsqueeze(-2).to(x.device) # B//2 , K , 1, L
            

        B = B // 2

        #self.merge_x(x), =  B//2,C,2*H*W
        #print('forward_core')
        #x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous()
        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
        x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
        xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ## projection
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # B, 4 *c , L
        Bs = Bs.float().view(B, K, -1, L) # B, 4, d_state , L
        
        

        Cs = Cs.float().view(B, K, -1, L)# B, 4, d_state , L
        Ds = self.Ds.float().view(-1) 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state) # Log a = 0~~~x 값인데.  e(0~x)   =  -1 ~ -x
        #dt_projs_bias = self.dt_projs_bias.float().view(-1)
        dt_projs_bias = self.dt_projs_bias.float()
        if dt_projs_bias is not None:
            #print(dts.shape , dt_projs_bias.shape)
            dts = dts + dt_projs_bias[...,None].float()
        if self.delta_softplus is True :
            dts = F.softplus(dts)
        if shift_mask != None:
            dts= dts*shift_mask #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L
            #Bs = Bs* shift_mask#  B//2, K, d_state , L *  B//2 , K , 1, L , b=0이면 현재입력 무시
            #Cs= Cs*shift_mask
            
        dts = dts.contiguous().float().view(B, -1, L) # B, 4 *d , L
        '''
        torch.Size([2048])   4* 512
        torch.Size([2048])
        torch.Size([4096])
        torch.Size([4096])
        '''
        out_y = self.selective_scan(
            xs, #u
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=False,
        ).view(B, K, -1, L)

        '''out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)'''



        assert out_y.dtype == torch.float

        '''
        def unmerge_x(self, x , H,W): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
        x=x.transpose(x,1,2).contiguous().view(B,-1,2,C) # B, L,2,C
        x= torch.cat( (x[:,:,0], x[:,:,1],0).view(B*2,H,W,C)).contiguous() # B,H,W,C

        return x
        '''
        #B==B//2
        y= out_y[:, 0]
        wh_y= out_y[:, 1] #B//2,C,2L

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = inv_y[:, 1]#B//2,C,2L
        
        wh_y = wh_y+ invwh_y # B//2,C,2L

        wh_y= self.unmerge_x(wh_y)# B,C,L
        wh_y = restore_z_ordered_tensor_4d(wh_y,(B*2, int(C),W,H), z_ordered_indices_wh)#.view(batch, channel, height, width)
        wh_y = torch.transpose(wh_y,2,3).contiguous()
        #.view(B, -1, W, H)
        

        y=  out_y[:, 0]+ inv_y[:, 0]
        y= self.unmerge_x(y) # B,C,L
        y = restore_z_ordered_tensor_4d(y,(B*2, int(C),H,W), z_ordered_indices)#.view(batch, channel, height, width)

        y= y+wh_y
        y= y.permute(0,2,3,1).contiguous()
 
        return y
    def window_partition(self,x, window_size):
        B, C, H, W = x.shape
        x=x.permute(0,2,3,1)
        x = x.view(B,H // window_size[0], window_size[0], W // window_size[1], window_size[1],C)
        windows = (
            x.permute(0, 1, 3,2,4,5).contiguous().view(-1, window_size[0],window_size[1],C)
        )
        return windows.permute(0,3,1,2)

    def window_reverse(self,windows, window_size, H, W):
        # nwB, N, C = windows.shape
        # windows = windows.view(-1, window_size[0], window_size[1], C)
        # B = int(nwB / (H * W / window_size[0] / window_size[1]))
        # x = windows.view(
        #     B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
        # )
        # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        nwB, w0,w1, C = windows.shape
        #windows = windows.view(-1, window_size[0], window_size[1], C)
        B = int(nwB / (H * W / window_size[0] / window_size[1]))

        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)

        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x



    def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # print(x.shape)
        y = self.forward_core(x, shift_mask )
        assert y.dtype == torch.float32
        # print(y.shape)

        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SS2D_zorder_local_shift(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            shift_size=0,
            window_size=[8,8],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        print('SS2D_zorder_local_shift')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.window_size=window_size
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn
        #self.selective_scan = selective_scan_mask_ref
        #self.selective_scan = selective_scan_mask_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.delta_softplus=True
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L
    def unmerge_x(self, x ): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
       # x = x.view(B,C,2,-1) # B,,C,2,L
       # x= torch.cat( (x[:,:,0], x[:,:,1]),0).view(B*2,C,L//2).contiguous() # B,H,W,C


        odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

        # 홀수, 짝수 순서로 배치
        x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

        #print((x-merged_tensor).max())
        return x

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # BNW,C,W0,W1
        #print(shift_mask.shape if shift_mask != None else 'None')  # B//2 NW ,1 ,L
        
        z_ordered_tensor, original_shape, z_ordered_indices = z_ordering_4d(x)
        z_ordered_tensor_wh, _, z_ordered_indices_wh = z_ordering_4d(torch.transpose(x, dim0=2, dim1=3).contiguous()) 
        
        #print("Z-ordered tensor:", z_ordered_tensor)
        x=z_ordered_tensor.view(B,C,H,W).contiguous()
        x_wh=z_ordered_tensor_wh.view(B,C,W,H).contiguous() 

        L = 2 * H * W #inter 
        K = 4 #4방향 
        
        if shift_mask != None:
            shift_mask=shift_mask.view(B,1,H,W)
            #print(shift_mask.shape)
            z_ordered_mask, _, _ = z_ordering_4d(shift_mask)
            z_ordered_mask_wh, _, _ = z_ordering_4d(torch.transpose(shift_mask, dim0=2, dim1=3).contiguous()) 
            z_ordered_mask=z_ordered_mask.view(B,1,H,W).contiguous()
            z_ordered_mask_wh=z_ordered_mask_wh.view(B,1,W,H).contiguous() 
            m_hwwh = torch.stack([self.merge_x(z_ordered_mask), self.merge_x(z_ordered_mask_wh)], dim=1).view(B//2, 2, 1, L) # B//2,2,C,2L,  horizon,vertical
            m_hwwh_reverse =  torch.flip(m_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
            ms = torch.cat([m_hwwh,m_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
            shift_mask = ms.float().view(B//2, -1, L).unsqueeze(-2).to(x.device) # B//2 , K , 1, L
            

        B = B // 2

        #self.merge_x(x), =  B//2,C,2*H*W
        #print('forward_core')
        #x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous()
        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
        x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
        xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ## projection
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # B, 4 *c , L
        Bs = Bs.float().view(B, K, -1, L) # B, 4, d_state , L
        
        

        Cs = Cs.float().view(B, K, -1, L)# B, 4, d_state , L
        Ds = self.Ds.float().view(-1) 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state) # Log a = 0~~~x 값인데.  e(0~x)   =  -1 ~ -x
        #dt_projs_bias = self.dt_projs_bias.float().view(-1)
        dt_projs_bias = self.dt_projs_bias.float()
        if dt_projs_bias is not None:
            #print(dts.shape , dt_projs_bias.shape)
            dts = dts + dt_projs_bias[...,None].float()
        if self.delta_softplus is True :
            dts = F.softplus(dts)
        if shift_mask != None:
            dts= dts*shift_mask #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L
            #Bs = Bs* shift_mask#  B//2, K, d_state , L *  B//2 , K , 1, L , b=0이면 현재입력 무시
            #Cs= Cs*shift_mask
            
        dts = dts.contiguous().float().view(B, -1, L) # B, 4 *d , L
        '''
        torch.Size([2048])   4* 512
        torch.Size([2048])
        torch.Size([4096])
        torch.Size([4096])
        '''
        out_y = self.selective_scan(
            xs, #u
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=False,
        ).view(B, K, -1, L)

        '''out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)'''



        assert out_y.dtype == torch.float

        '''
        def unmerge_x(self, x , H,W): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
        x=x.transpose(x,1,2).contiguous().view(B,-1,2,C) # B, L,2,C
        x= torch.cat( (x[:,:,0], x[:,:,1],0).view(B*2,H,W,C)).contiguous() # B,H,W,C

        return x
        '''
        #B==B//2
        y= out_y[:, 0]
        wh_y= out_y[:, 1] #B//2,C,2L

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = inv_y[:, 1]#B//2,C,2L
        
        wh_y = wh_y+ invwh_y # B//2,C,2L

        wh_y= self.unmerge_x(wh_y)# B,C,L
        wh_y = restore_z_ordered_tensor_4d(wh_y,(B*2, int(C),W,H), z_ordered_indices_wh)#.view(batch, channel, height, width)
        wh_y = torch.transpose(wh_y,2,3).contiguous()
        #.view(B, -1, W, H)
        

        y=  out_y[:, 0]+ inv_y[:, 0]
        y= self.unmerge_x(y) # B,C,L
        y = restore_z_ordered_tensor_4d(y,(B*2, int(C),H,W), z_ordered_indices)#.view(batch, channel, height, width)

        y= y+wh_y
        y= y.permute(0,2,3,1).contiguous()
 
        return y
    def window_partition(self,x, window_size):
        B, C, H, W = x.shape
        x=x.permute(0,2,3,1)
        x = x.view(B,H // window_size[0], window_size[0], W // window_size[1], window_size[1],C)
        windows = (
            x.permute(0, 1, 3,2,4,5).contiguous().view(-1, window_size[0],window_size[1],C)
        )
        return windows.permute(0,3,1,2)


    def window_reverse(self,windows, window_size, H, W):
        # nwB, N, C = windows.shape
        # windows = windows.view(-1, window_size[0], window_size[1], C)
        # B = int(nwB / (H * W / window_size[0] / window_size[1]))
        # x = windows.view(
        #     B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
        # )
        # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        nwB, w0,w1, C = windows.shape
        #windows = windows.view(-1, window_size[0], window_size[1], C)
        B = int(nwB / (H * W / window_size[0] / window_size[1]))

        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)

        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x



    def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
 
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        
        x = self.window_partition(x, self.window_size) # B*NW,C,H,W
        # nwB = x.shape[0]
        # # x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1)
        # # print(x.shape)
        y = self.forward_core(x, shift_mask ) #b,h,w,c
        assert y.dtype == torch.float32 
        # print(y.shape)
        y = self.out_norm(y)
        y= self.window_reverse(y,self.window_size,H,W)#B,H,W,C
        y = y * F.silu(z)
        
        out = self.out_proj(y)#B,H,W,C

        if self.dropout is not None:
            out = self.dropout(out)
        return out


########################################################################################################################################################################################################################
########################################################################################################################################################################################################################

class SS2D_hilbert_shift(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            shift_size=0,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        print('SS2D_zorder_shift')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn
        #self.selective_scan = selective_scan_mask_ref
        #self.selective_scan = selective_scan_mask_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.delta_softplus=True

        # B, C, H, W = tensor.shape
        # L = H * W  # 픽셀 수
        # 힐베르트 곡선의 차원과 단계 설정
        p = int(np.log2(8))  # 힐베르트 곡선의 단계 (order)
        n = 2  # 2차원

        H,W=8,8

        # 힐베르트 곡선 객체 생성
        hilbert_curve = HilbertCurve(p, n)

        # 힐베르트 곡선의 전체 좌표 계산
        coords = []
        for y in range(H):
            for x in range(W):
                coords.append((x, y))

        # 각 좌표에 대한 힐베르트 인덱스 계산
        hilbert_indices = []
        for coord in coords:
            x, y = coord
            # 힐베르트 곡선의 크기에 맞게 좌표 조정
            hilbert_index = hilbert_curve.distance_from_point([x, y])
            hilbert_indices.append(hilbert_index)

        # 힐베르트 인덱스에 따라 정렬
        hilbert_indices = np.array(hilbert_indices)
        self.sorted_indices = np.argsort(hilbert_indices)
        # 역순서 인덱스 계산
        self.inverse_indices = np.argsort( self.sorted_indices)

        # 입력 텐서를 힐베르트 순서로 재배열
        #tensor_flat = tensor.view(B,  C, -1)  # (B,K, C, H*W)
        #hilbert_tensor = tensor_flat[: , : , sorted_indices]  # (B,K, C,L)

        
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L
    def unmerge_x(self, x ): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
       # x = x.view(B,C,2,-1) # B,,C,2,L
       # x= torch.cat( (x[:,:,0], x[:,:,1]),0).view(B*2,C,L//2).contiguous() # B,H,W,C


        odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

        # 홀수, 짝수 순서로 배치
        x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

        #print((x-merged_tensor).max())
        return x

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # BNW,C,W0,W1
        #print(shift_mask.shape if shift_mask != None else 'None')  # B//2 NW ,1 ,L
        
        #x=x.permute(0,2,3,1) # 
        z_ordered_tensor = apply_hilbert_curve_2d(x,self.sorted_indices)
        z_ordered_tensor_wh = apply_hilbert_curve_2d(torch.transpose(x, dim0=2, dim1=3).contiguous(),self.sorted_indices) 
        
        #print("Z-ordered tensor:", z_ordered_tensor)
        x=z_ordered_tensor.view(B,C,H,W).contiguous()
        x_wh=z_ordered_tensor_wh.view(B,C,W,H).contiguous() 

        L = 2 * H * W #inter 
        K = 4 #4방향 
        
        if shift_mask != None:
            shift_mask=shift_mask.view(B,1,H,W)
            #print(shift_mask.shape)
            z_ordered_mask= apply_hilbert_curve_2d(shift_mask,self.sorted_indices)
            z_ordered_mask_wh= apply_hilbert_curve_2d(torch.transpose(shift_mask, dim0=2, dim1=3).contiguous(),self.sorted_indices) 
            z_ordered_mask=z_ordered_mask.view(B,1,H,W).contiguous()
            z_ordered_mask_wh=z_ordered_mask_wh.view(B,1,W,H).contiguous() 
            m_hwwh = torch.stack([self.merge_x(z_ordered_mask), self.merge_x(z_ordered_mask_wh)], dim=1).view(B//2, 2, 1, L) # B//2,2,C,2L,  horizon,vertical
            m_hwwh_reverse =  torch.flip(m_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
            ms = torch.cat([m_hwwh,m_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
            shift_mask = ms.float().view(B//2, -1, L).unsqueeze(-2).to(x.device) # B//2 , K , 1, L
            

        B = B // 2

        #self.merge_x(x), =  B//2,C,2*H*W
        #print('forward_core')
        #x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous()
        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
        x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
        xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ## projection
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # B, 4 *c , L
        Bs = Bs.float().view(B, K, -1, L) # B, 4, d_state , L
        
        

        Cs = Cs.float().view(B, K, -1, L)# B, 4, d_state , L
        Ds = self.Ds.float().view(-1) 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        #dt_projs_bias = self.dt_projs_bias.float().view(-1)
        dt_projs_bias = self.dt_projs_bias.float()
        if dt_projs_bias is not None:
            #print(dts.shape , dt_projs_bias.shape)
            dts = dts + dt_projs_bias[...,None].float()
        if self.delta_softplus is True :
            dts = F.softplus(dts)
        if shift_mask != None:
            dts= dts*shift_mask #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L
            #Bs = Bs* shift_mask#  B//2, K, d_state , L *  B//2 , K , 1, L , b=0이면 현재입력 무시
            #Cs= Cs*shift_mask
            
        dts = dts.contiguous().float().view(B, -1, L) # B, 4 *d , L
        '''
        torch.Size([2048])   4* 512
        torch.Size([2048])
        torch.Size([4096])
        torch.Size([4096])
        '''
        out_y = self.selective_scan(
            xs, #u
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=False,
        ).view(B, K, -1, L)

        '''out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)'''



        assert out_y.dtype == torch.float

        '''
        def unmerge_x(self, x , H,W): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
        x=x.transpose(x,1,2).contiguous().view(B,-1,2,C) # B, L,2,C
        x= torch.cat( (x[:,:,0], x[:,:,1],0).view(B*2,H,W,C)).contiguous() # B,H,W,C

        return x
        '''
        #B==B//2
        y= out_y[:, 0]
        wh_y= out_y[:, 1] #B//2,C,2L

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = inv_y[:, 1]#B//2,C,2L
        
        wh_y = wh_y+ invwh_y # B//2,C,2L

        wh_y= self.unmerge_x(wh_y)# B,C,L
        wh_y = reverse_hilbert_curve_2d(wh_y, self.inverse_indices , W,H)#.view(batch, channel, height, width)
        wh_y = torch.transpose(wh_y,2,3).contiguous()
        #.view(B, -1, W, H)
        

        y=  out_y[:, 0]+ inv_y[:, 0]
        y= self.unmerge_x(y) # B,C,L
        y = reverse_hilbert_curve_2d(y,self.inverse_indices,H,W)#.view(batch, channel, height, width)

        y= y+wh_y
        y= y.permute(0,2,3,1).contiguous()
 
        return y

    def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # print(x.shape)
        y = self.forward_core(x, shift_mask )
        assert y.dtype == torch.float32
        # print(y.shape)

        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out

########################################################################################################################################################################################################################
########################################################################################################################################################################################################################


class SS2D_hilbert_shift2(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            shift_size=0,
            window_size=[8,8],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        print('SS2D_hilbert_shift2')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.window_size=window_size
        print(window_size)
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn
        #self.selective_scan = selective_scan_mask_ref
        #self.selective_scan = selective_scan_mask_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.delta_softplus=True

        # B, C, H, W = tensor.shape
        # L = H * W  # 픽셀 수
        # 힐베르트 곡선의 차원과 단계 설정
        p = int(np.log2(window_size[0]))  # 힐베르트 곡선의 단계 (order)
        n = 2  # 2차원

        H,W=window_size[0],window_size[1]

        # 힐베르트 곡선 객체 생성
        hilbert_curve = HilbertCurve(p, n)

        # 힐베르트 곡선의 전체 좌표 계산
        coords = []
        for y in range(H):
            for x in range(W):
                coords.append((x, y))

        # 각 좌표에 대한 힐베르트 인덱스 계산
        hilbert_indices = []
        for coord in coords:
            x, y = coord
            # 힐베르트 곡선의 크기에 맞게 좌표 조정
            hilbert_index = hilbert_curve.distance_from_point([x, y])
            hilbert_indices.append(hilbert_index)

        # 힐베르트 인덱스에 따라 정렬
        hilbert_indices = np.array(hilbert_indices)
        self.sorted_indices = np.argsort(hilbert_indices)
        # 역순서 인덱스 계산
        self.inverse_indices = np.argsort( self.sorted_indices)

        # 입력 텐서를 힐베르트 순서로 재배열
        #tensor_flat = tensor.view(B,  C, -1)  # (B,K, C, H*W)
        #hilbert_tensor = tensor_flat[: , : , sorted_indices]  # (B,K, C,L)

        
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L
    def unmerge_x(self, x ): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
       # x = x.view(B,C,2,-1) # B,,C,2,L
       # x= torch.cat( (x[:,:,0], x[:,:,1]),0).view(B*2,C,L//2).contiguous() # B,H,W,C


        odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

        # 홀수, 짝수 순서로 배치
        x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

        #print((x-merged_tensor).max())
        return x

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # BNW,C,W0,W1
        #print(shift_mask.shape if shift_mask != None else 'None')  # B//2 NW ,1 ,L
        
        #x=x.permute(0,2,3,1) # 
        z_ordered_tensor = apply_hilbert_curve_2d(x,self.sorted_indices)
        z_ordered_tensor_wh = apply_hilbert_curve_2d(torch.transpose(x, dim0=2, dim1=3).contiguous(),self.sorted_indices) 
        
        #print("Z-ordered tensor:", z_ordered_tensor)
        x=z_ordered_tensor.view(B,C,H,W).contiguous()
        x_wh=z_ordered_tensor_wh.view(B,C,W,H).contiguous() 

        L = 2 * H * W #inter 
        K = 4 #4방향 
        
        if shift_mask != None:
            shift_mask=shift_mask.view(B,1,H,W)
            #print(shift_mask.shape)
            z_ordered_mask= apply_hilbert_curve_2d(shift_mask,self.sorted_indices)
            z_ordered_mask_wh= apply_hilbert_curve_2d(torch.transpose(shift_mask, dim0=2, dim1=3).contiguous(),self.sorted_indices) 
            z_ordered_mask=z_ordered_mask.view(B,1,H,W).contiguous()
            z_ordered_mask_wh=z_ordered_mask_wh.view(B,1,W,H).contiguous() 
            m_hwwh = torch.stack([self.merge_x(z_ordered_mask), self.merge_x(z_ordered_mask_wh)], dim=1).view(B//2, 2, 1, L) # B//2,2,C,2L,  horizon,vertical
            m_hwwh_reverse =  torch.flip(m_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
            ms = torch.cat([m_hwwh,m_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
            shift_mask = ms.float().view(B//2, -1, L).unsqueeze(-2).to(x.device) # B//2 , K , 1, L
            

        B = B // 2

        #self.merge_x(x), =  B//2,C,2*H*W
        #print('forward_core')
        #x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous()
        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
        x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
        xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ## projection
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # B, 4 *c , L
        Bs = Bs.float().view(B, K, -1, L) # B, 4, d_state , L
        
        

        Cs = Cs.float().view(B, K, -1, L)# B, 4, d_state , L
        Ds = self.Ds.float().view(-1) 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        #dt_projs_bias = self.dt_projs_bias.float().view(-1)
        dt_projs_bias = self.dt_projs_bias.float()
        if dt_projs_bias is not None:
            #print(dts.shape , dt_projs_bias.shape)
            dts = dts + dt_projs_bias[...,None].float()
        if self.delta_softplus is True :
            dts = F.softplus(dts)
        if shift_mask != None:
            dts= dts*shift_mask #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L
            #Bs = Bs* shift_mask#  B//2, K, d_state , L *  B//2 , K , 1, L , b=0이면 현재입력 무시
            #Cs= Cs*shift_mask
            
        dts = dts.contiguous().float().view(B, -1, L) # B, 4 *d , L
        '''
        torch.Size([2048])   4* 512
        torch.Size([2048])
        torch.Size([4096])
        torch.Size([4096])
        '''
        out_y = self.selective_scan(
            xs, #u
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=False,
        ).view(B, K, -1, L)

        '''out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)'''



        assert out_y.dtype == torch.float

        '''
        def unmerge_x(self, x , H,W): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
        x=x.transpose(x,1,2).contiguous().view(B,-1,2,C) # B, L,2,C
        x= torch.cat( (x[:,:,0], x[:,:,1],0).view(B*2,H,W,C)).contiguous() # B,H,W,C

        return x
        '''
        #B==B//2
        y= out_y[:, 0]
        wh_y= out_y[:, 1] #B//2,C,2L

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = inv_y[:, 1]#B//2,C,2L
        
        wh_y = wh_y+ invwh_y # B//2,C,2L

        wh_y= self.unmerge_x(wh_y)# B,C,L
        wh_y = reverse_hilbert_curve_2d(wh_y, self.inverse_indices , W,H)#.view(batch, channel, height, width)
        wh_y = torch.transpose(wh_y,2,3).contiguous()
        #.view(B, -1, W, H)
        

        y=  out_y[:, 0]+ inv_y[:, 0]
        y= self.unmerge_x(y) # B,C,L
        y = reverse_hilbert_curve_2d(y,self.inverse_indices,H,W)#.view(batch, channel, height, width)

        y= y+wh_y
        y= y.permute(0,2,3,1).contiguous()
 
        return y


    def window_partition(self,x, window_size):
        B, C, H, W = x.shape
        x=x.permute(0,2,3,1)
        x = x.view(B,H // window_size[0], window_size[0], W // window_size[1], window_size[1],C)
        windows = (
            x.permute(0, 1, 3,2,4,5).contiguous().view(-1, window_size[0],window_size[1],C)
        )
        return windows.permute(0,3,1,2)
    def window_reverse(self,windows, window_size, H, W):
        # nwB, N, C = windows.shape
        # windows = windows.view(-1, window_size[0], window_size[1], C)
        # B = int(nwB / (H * W / window_size[0] / window_size[1]))
        # x = windows.view(
        #     B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
        # )
        # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        nwB, w0,w1, C = windows.shape
        #windows = windows.view(-1, window_size[0], window_size[1], C)
        B = int(nwB / (H * W / window_size[0] / window_size[1]))

        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)

        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x



    def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
 
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        
        x = self.window_partition(x, self.window_size) # B*NW,C,H,W
        # nwB = x.shape[0]
        # # x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1)
        # # print(x.shape)
        y = self.forward_core(x, shift_mask ) #b,h,w,c
        assert y.dtype == torch.float32 
        # print(y.shape)
        y = self.out_norm(y)
        y= self.window_reverse(y,self.window_size,H,W)#B,H,W,C
        y = y * F.silu(z)
        
        out = self.out_proj(y)#B,H,W,C

        if self.dropout is not None:
            out = self.dropout(out)
        return out

########################################################################################################################################################################################################################
########################################################################################################################################################################################################################


class SS2D_hilbert_3d_shift(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            shift_size=0,
            window_size=[4,4],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        print('SS2D_hilbert_3d_shift')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.window_size=window_size
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn
        #self.selective_scan = selective_scan_mask_ref
        #self.selective_scan = selective_scan_mask_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.delta_softplus=True
        
      
        self.coords= []

        w,h,d=8,8,2
        n = w*h*d

        for idx in range(n):
            self.coords.append(gilbert_d2xyz(idx,w,h,d))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L
    def unmerge_x(self, x ): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
       # x = x.view(B,C,2,-1) # B,,C,2,L
       # x= torch.cat( (x[:,:,0], x[:,:,1]),0).view(B*2,C,L//2).contiguous() # B,H,W,C


        odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

        # 홀수, 짝수 순서로 배치
        x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

        #print((x-merged_tensor).max())
        return x

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # BNW,C,W0,W1
        #print(shift_mask.shape if shift_mask != None else 'None')  # B//2 NW ,1 ,L
        sc=x
        #x=x.permute(0,2,3,1) # 

        x = torch.stack((x[:B//2],x[B//2:]),1) # B,T,C,H,W

        h_ordered_tensor = apply_hilbert_curve_3d_vectorized(x,self.coords)#b,c.2L
        #x_wh= (torch.transpose(x, dim0=-2, dim1=-1).contiguous())
        x_wh= x = torch.stack((sc[B//2:],sc[:B//2]),1) # B,T,C,H,W
        h_ordered_tensor_wh = apply_hilbert_curve_3d_vectorized(x_wh,self.coords)#b,c.2L
        
        #print("Z-ordered tensor:", z_ordered_tensor)

        L = 2 * H * W #inter 
        K = 4 #4방향 
        
        if shift_mask != None:
            sm=shift_mask
            shift_mask=torch.stack((sm[:B//2],sm[:B//2]),1).view(B//2,2,1,H,W)
            shift_mask_wh=torch.stack((sm[B//2:],sm[B//2:]),1).view(B//2,2,1,H,W)
            #shift_mask_wh= torch.transpose(shift_mask, dim0=-2, dim1=-1).contiguous()
            #shift_mask=shift_mask[:B//2].view(B//2,1,H,W)
            #print(shift_mask.shape)
            h_ordered_mask = apply_hilbert_curve_3d_vectorized(shift_mask,self.coords) #B,c,2l
            h_ordered_mask_wh = apply_hilbert_curve_3d_vectorized(shift_mask_wh,self.coords) 
            # z_ordered_mask=z_ordered_mask.view(B,1,H,W).contiguous()
            # z_ordered_mask_wh=z_ordered_mask_wh.view(B,1,W,H).contiguous() 
            m_hwwh = torch.stack([h_ordered_mask, h_ordered_mask_wh ], dim=1) # B//2,2,C,2L,  horizon,vertical
            m_hwwh_reverse =  torch.flip(m_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
            ms = torch.cat([m_hwwh,m_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
            #shift_mask = ms.float().view(B//2, -1, L).unsqueeze(-2).to(x.device) # B//2 , K , 1, L
            shift_mask = ms.to(x.device) # B//2 , K , 1, L
            #print(shift_mask.shape)
            

        B = B // 2

        #self.merge_x(x), =  B//2,C,2*H*W
        #print('forward_core')
        #x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous() 

       
        x_hwwh = torch.stack([h_ordered_tensor, h_ordered_tensor_wh], dim=1) #B//2,2,C,2L
        #.view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
        x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
        xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ## projection
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # B, 4 *c , L
        Bs = Bs.float().view(B, K, -1, L) # B, 4, d_state , L
        
        

        Cs = Cs.float().view(B, K, -1, L)# B, 4, d_state , L
        Ds = self.Ds.float().view(-1) 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        #dt_projs_bias = self.dt_projs_bias.float().view(-1)
        dt_projs_bias = self.dt_projs_bias.float()
        if dt_projs_bias is not None:
            #print(dts.shape , dt_projs_bias.shape)
            dts = dts + dt_projs_bias[...,None].float()
        if self.delta_softplus is True :
            dts = F.softplus(dts)
        if shift_mask != None:
            dts= dts*shift_mask #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L
            #Bs = Bs* shift_mask#  B//2, K, d_state , L *  B//2 , K , 1, L , b=0이면 현재입력 무시
            #Cs= Cs*shift_mask
            
        dts = dts.contiguous().float().view(B, -1, L) # B, 4 *d , L

        '''
        torch.Size([2048])   4* 512
        torch.Size([2048])
        torch.Size([4096])
        torch.Size([4096])
        '''
        out_y = self.selective_scan(
            xs, #u
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=False,
        ).view(B, K, -1, L)

        '''out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)'''



        assert out_y.dtype == torch.float
        #print(out_y.device)

        '''
        def unmerge_x(self, x , H,W): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
        x=x.transpose(x,1,2).contiguous().view(B,-1,2,C) # B, L,2,C
        x= torch.cat( (x[:,:,0], x[:,:,1],0).view(B*2,H,W,C)).contiguous() # B,H,W,C

        return x
        '''
        #B==B//2
        y= out_y[:, 0]
        wh_y= out_y[:, 1] #B//2,C,2L

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = inv_y[:, 1]#B//2,C,2L
        
        wh_y = wh_y+ invwh_y # B//2,C,2L

        #wh_y= self.unmerge_x(wh_y)# B,C,L
        wh_y = restore_hilbert_curve_3d_vectorized(wh_y, self.coords , x.shape   )#.view(batch, t, channel, height, width)
        wh_y = torch.transpose(wh_y,-2,-1).contiguous()
        #.view(B, -1, W, H)
        

        y=  out_y[:, 0]+ inv_y[:, 0]
        #y= self.unmerge_x(y) # B,C,L
        y = restore_hilbert_curve_3d_vectorized(y,self.coords, x.shape )#.view(batch,t, channel, height, width)

        y= y + wh_y
        y= y.view(B*2,C,H,W)
        y= y.permute(0,2,3,1).contiguous()
 
        return y

    def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # print(x.shape)
        y = self.forward_core(x, shift_mask )
        assert y.dtype == torch.float32
        # print(y.shape)

        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out
######################################################################################################################################################################################################
######################################################################################################################################################################################################
class SS2D_hilbert_shift_rot(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            shift_size=0,
            window_size=[8,8],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        print('SS2D_hilbert_shift_rot')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.window_size=window_size
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs)
        )
        self.k=4
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.k, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.k, merge=True)

        self.selective_scan = selective_scan_fn
        #self.selective_scan = selective_scan_mask_ref
        #self.selective_scan = selective_scan_mask_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.delta_softplus=True

        
        
        p = int(np.log2(window_size[0]))  # 힐베르트 곡선의 단계 (order)
        n = 2  # 2차원

        H,W=window_size[0],window_size[1] #window size 

        # 힐베르트 곡선 객체 생성
        hilbert_curve = HilbertCurve(p, n)

        # 힐베르트 곡선의 전체 좌표 계산
        coords = []
        for y in range(H):
            for x in range(W):
                coords.append((x, y))

        # 각 좌표에 대한 힐베르트 인덱스 계산
        hilbert_indices = []
        for coord in coords:
            x, y = coord
            # 힐베르트 곡선의 크기에 맞게 좌표 조정
            hilbert_index = hilbert_curve.distance_from_point([x, y])
            hilbert_indices.append(hilbert_index)

        # 힐베르트 인덱스에 따라 정렬
        hilbert_indices = np.array(hilbert_indices)
        self.sorted_indices = np.argsort(hilbert_indices)
        # 역순서 인덱스 계산
        self.inverse_indices = np.argsort( self.sorted_indices)


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj


    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L
    def unmerge_x(self, x ): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
       # x = x.view(B,C,2,-1) # B,,C,2,L
       # x= torch.cat( (x[:,:,0], x[:,:,1]),0).view(B*2,C,L//2).contiguous() # B,H,W,C


        odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

        # 홀수, 짝수 순서로 배치
        x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

        #print((x-merged_tensor).max())
        return x

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # BNW,C,W0,W1
        K = self.k #4방향 
        #print(shift_mask.shape if shift_mask != None else 'None')  # B//2 NW ,1 ,L
        L = H * W #inter 

        #x=x.permute(0,2,3,1) # 

        x_0 = x
        x_1 = torch.rot90(x, k=1, dims=(2, 3)) #시계반대방향 
        x_2 = torch.rot90(x, k=2, dims=(2, 3))
        x_3 = torch.rot90(x, k=3, dims=(2, 3))

        x= torch.stack([x_0,x_1,x_2,x_3],1) # B,4C,H,W
        x=x.view(B,K*C,H,W)
        h_x  = apply_hilbert_curve_2d(x,self.sorted_indices)#B,4c,L
        #h_1, z_ordered_indices1 = apply_hilbert_curve(x_1).contiguous()
        #h_2, z_ordered_indices2 = apply_hilbert_curve(x_2).contiguous()
        #h_3, z_ordered_indices3 = apply_hilbert_curve(x_3).contiguous() 


        #z_ordered_tensor_wh, z_ordered_indices_wh = apply_hilbert_curve(torch.transpose(x, dim0=2, dim1=3).contiguous()) 
        
        #print("Z-ordered tensor:", z_ordered_tensor)
        #x=h_x.view(B,-1,H,W).contiguous()

        
        if shift_mask != None:
            shift_mask=shift_mask.view(B,1,H,W).to(x.device)
            #print(shift_mask.shape)
            s_1=torch.rot90(shift_mask, k=1, dims=(-2, -1))
            s_2=torch.rot90(shift_mask, k=2, dims=(-2, -1))
            s_3=torch.rot90(shift_mask, k=3, dims=(-2, -1))
            shift_mask  = torch.cat([shift_mask,s_1,s_2,s_3],1).contiguous() # B,4,H,W
            shift_mask  = apply_hilbert_curve_2d(shift_mask,self.sorted_indices) #B,4,L
            #shift_mask_inv= torch.flip(shift_mask,dims=[-1])
            #shift_mask=torch.cat((shift_mask,shift_mask_inv),1) #B,K,1,L
            shift_mask=shift_mask.view(B//2,K,1,-1)
        #self.merge_x(x), =  B//2,C,2*H*W
        #print('forward_core')
        #x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous()
        #x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
        #x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
        #xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
        
        #x=x.permute(0,1,3,2) #B,K,L,C #BKCL

        # 2L로 만드는 과정 

        h_x=h_x.view(B,K*C,H,W)

        xs=self.merge_x(h_x)

        L = 2 * H * W #inter 
        B = B // 2
       

        #xs_inv = torch.flip(xs,dims=[-1])
        #xs=torch.cat((xs,xs_inv),1) #B//2,K,C,2L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ## projection
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # B, 4 *c , L
        Bs = Bs.float().view(B, K, -1, L) # B, 4, d_state , L
        
        Cs = Cs.float().view(B, K, -1, L)# B, 4, d_state , L
        Ds = self.Ds.float().view(-1) 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        #dt_projs_bias = self.dt_projs_bias.float().view(-1)
        dt_projs_bias = self.dt_projs_bias.float()
        
        if dt_projs_bias is not None:
            #print(dts.shape , dt_projs_bias.shape)
            dts = dts + dt_projs_bias[...,None].float()
        if self.delta_softplus is True :
            dts = F.softplus(dts)
        if shift_mask != None:
            dts= dts*shift_mask #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L
            #Bs = Bs* shift_mask#  B//2, K, d_state , L *  B//2 , K , 1, L , b=0이면 현재입력 무시
            #Cs= Cs*shift_mask
            
        dts = dts.contiguous().float().view(B, -1, L) # B, 4 *d , L
        '''
        torch.Size([2048])   4* 512
        torch.Size([2048])
        torch.Size([4096])
        torch.Size([4096])
        '''
        out_y = self.selective_scan(
            xs, #u
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=False,
        ).view(B, K, -1, L)

        '''out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)'''



        assert out_y.dtype == torch.float

        '''
        def unmerge_x(self, x , H,W): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
        x=x.transpose(x,1,2).contiguous().view(B,-1,2,C) # B, L,2,C
        x= torch.cat( (x[:,:,0], x[:,:,1],0).view(B*2,H,W,C)).contiguous() # B,H,W,C

        return x
        '''
        #B==B//2


        out_y=out_y[:,:4]
        #+ torch.flip(out_y[:,4:],dims=[-1]) #B,K,C,L
        out_y= out_y.view(B,K*C,L) #,B,4,C,L
        # print(out_y.shape)
        # print(out_y[:,:,0::2].shape)
        # print(out_y[:,:,1::2].shape)
        #out_y= torch.cat((out_y[:,:,0::2],out_y[:,:,1::2]),0)
        
        out_y=self.unmerge_x(out_y)
        B=B*2
        L=L//2
        out_y = reverse_hilbert_curve_2d(out_y,self.inverse_indices,H,W).view(B,K,C,H,W)
        #.view(B,4,C,L)#.view(batch,channel, height, width)


        y0= out_y[:, 0]
        y1= out_y[:, 1]
        y2= out_y[:, 2]
        y3= out_y[:, 3]
        
        y1=torch.rot90(y1, k=-1, dims=(-2, -1))
        y2=torch.rot90(y2, k=-2, dims=(-2, -1))
        y3=torch.rot90(y3, k=-3, dims=(-2, -1))
        
        y=y0+y3+y1+y2 #,B,C,H,W
        #y= y.view(B,-1,L)
        # odd_elements = y[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        # even_elements = y[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)
        # y = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()


        # 홀수, 짝수 순서로 배치

        y= y.permute(0,2,3,1).contiguous()
 
        return y


    def window_partition(self,x, window_size):
        B, C, H, W = x.shape
        x=x.permute(0,2,3,1)
        x = x.view(B,H // window_size[0], window_size[0], W // window_size[1], window_size[1],C)
        windows = (
            x.permute(0, 1, 3,2,4,5).contiguous().view(-1, window_size[0],window_size[1],C)
        )
        return windows.permute(0,3,1,2)

    def window_reverse(self,windows, window_size, H, W):
        # nwB, N, C = windows.shape
        # windows = windows.view(-1, window_size[0], window_size[1], C)
        # B = int(nwB / (H * W / window_size[0] / window_size[1]))
        # x = windows.view(
        #     B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
        # )
        # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        nwB, w0,w1, C = windows.shape
        #windows = windows.view(-1, window_size[0], window_size[1], C)
        B = int(nwB / (H * W / window_size[0] / window_size[1]))

        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)

        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x



    def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
 
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        
        x = self.window_partition(x, self.window_size) # B*NW,C,H,W
        # nwB = x.shape[0]
        # # x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1)
        # # print(x.shape)
        y = self.forward_core(x, shift_mask ) #b,h,w,c
        assert y.dtype == torch.float32 
        # print(y.shape)
        y = self.out_norm(y)
        y= self.window_reverse(y,self.window_size,H,W)#B,H,W,C
        y = y * F.silu(z)
        
        out = self.out_proj(y)#B,H,W,C

        if self.dropout is not None:
            out = self.dropout(out)
        return out



class SS2D_hilbert_3d_shift_rot(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            shift_size=0,
            window_size=[8,8],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        print('SS2D_hilbert_3d_shift_rot')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.window_size=window_size
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        )
        
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.k=4
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.k, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.k , merge=True)

        self.selective_scan = selective_scan_fn


        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.delta_softplus=True
        
      
        self.coords= []

        h,w,d=window_size[-2],window_size[-1],2
        n = w*h*d
        print(window_size)

        for idx in range(n):
            self.coords.append(gilbert_d2xyz(idx,w,h,d))


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L
    def unmerge_x(self, x ): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
       # x = x.view(B,C,2,-1) # B,,C,2,L
       # x= torch.cat( (x[:,:,0], x[:,:,1]),0).view(B*2,C,L//2).contiguous() # B,H,W,C


        odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

        # 홀수, 짝수 순서로 배치
        x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

        #print((x-merged_tensor).max())
        return x

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # BNW,C,W0,W1
    

        L = 2 * H * W #inter 
        K = self.k #4방향 

        x = torch.stack((x[:B//2],x[B//2:]),1) # B,T,C,H,W
        
        x0 = x
        x1=torch.rot90(x, k=1, dims=(-2, -1))
        x2=torch.rot90(x, k=2, dims=(-2, -1))
        x3=torch.rot90(x, k=3, dims=(-2, -1))

        xs= torch.cat([x0,x1,x2,x3],2) # B,T,4C,H,W, T=2 :  B//2,2,4c,h,w
        h_ordered_tensor = apply_hilbert_curve_3d_vectorized(xs,self.coords) #B//2,4C,2L
        h_ordered_tensor=h_ordered_tensor.view(B//2,4,C,L)#b//2,4,c,2L
    
        if shift_mask != None:
            shift_mask=shift_mask.view(B,1,H,W).to(x.device)
            shift_mask = torch.stack((shift_mask[:B//2],shift_mask[B//2:]),1) # B,T,C,H,W
            x1=torch.rot90(shift_mask, k=1, dims=(-2, -1))
            x2=torch.rot90(shift_mask, k=2, dims=(-2, -1))
            x3=torch.rot90(shift_mask, k=3, dims=(-2, -1))
            shift_mask= torch.cat([shift_mask,x1,x2,x3],2) # B,T,4C,H,W, T=2
            h_ordered_mask = apply_hilbert_curve_3d_vectorized(shift_mask,self.coords) #B//2,4c,2l , c=1
            shift_mask = h_ordered_mask.view(B//2,4,1,L)#B,4,1,L        
    
        B = B // 2


        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ## projection
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # B, 4 *c , L
        Bs = Bs.float().view(B, K, -1, L) # B, 4, d_state , L
        Cs = Cs.float().view(B, K, -1, L)# B, 4, d_state , L
        Ds = self.Ds.float().view(-1) 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float()

        if dt_projs_bias is not None:
            dts = dts + dt_projs_bias[...,None].float()
        if self.delta_softplus is True :
            dts = F.softplus(dts)
        if shift_mask != None:
            dts= dts*shift_mask #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L
            
        dts = dts.contiguous().float().view(B, -1, L) # B, 4 *d , L

        out_y = self.selective_scan(
            xs, #u
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        out_y=out_y.view(B,-1,L)# B,4,c,2L -> B,4C,2L
        
        L=L//2
        out_y = restore_hilbert_curve_3d_vectorized(out_y, self.coords , x.shape ).view(B,2,4,C,H,W)#.view(batch, t, channel, height, width)
        B=B*2

        y= out_y[:,:, 0]
        y1=out_y[:,:, 1]
        y2=out_y[:,:, 2]
        y3=out_y[:,:, 3]
        
        y1=torch.rot90(y1, k=-1, dims=(-2, -1))
        y2=torch.rot90(y2, k=-2, dims=(-2, -1))
        y3=torch.rot90(y3, k=-3, dims=(-2, -1))
        
        y=y+y3+y1+y2 #,B//2 ,T, C,H,W
        #y= y.view(B,C,H,W)
        y= torch.cat((y[:,0],y[:,1]),0)
        y= y.permute(0,2,3,1).contiguous()


 
        return y

    def window_partition(self,x, window_size):
        B, C, H, W = x.shape
        x=x.permute(0,2,3,1)
        x = x.view(B,H // window_size[0], window_size[0], W // window_size[1], window_size[1],C)
        windows = (
            x.permute(0, 1, 3,2,4,5).contiguous().view(-1, window_size[0],window_size[1],C)
        )
        return windows.permute(0,3,1,2)
    def window_reverse(self,windows, window_size, H, W):
        # nwB, N, C = windows.shape
        # windows = windows.view(-1, window_size[0], window_size[1], C)
        # B = int(nwB / (H * W / window_size[0] / window_size[1]))
        # x = windows.view(
        #     B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
        # )
        # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        nwB, w0,w1, C = windows.shape
        #windows = windows.view(-1, window_size[0], window_size[1], C)
        B = int(nwB / (H * W / window_size[0] / window_size[1]))

        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)

        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x



    def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
 
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        
        x = self.window_partition(x, self.window_size) # B*NW,C,H,W
        # nwB = x.shape[0]
        # # x_win=x_win.view(nwB,self.window_size[0],self.window_size[1],-1)
        # # print(x.shape)
        y = self.forward_core(x, shift_mask ) #b,h,w,c
        assert y.dtype == torch.float32 
        # print(y.shape)
        y = self.out_norm(y)
        y= self.window_reverse(y,self.window_size,H,W)#B,H,W,C
        y = y * F.silu(z)
        
        out = self.out_proj(y)#B,H,W,C

        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SS2D_localmamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            window_size=[8,8],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.window_size=window_size
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn
        #self.selective_scan = selective_scan_ref

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x =  x.reshape(B,C, H//self.window_size[0], self.window_size[0], W//self.window_size[1] , self.window_size[1]).permute(0,1,2,4,3,5).contiguous()
        x=x.view(B,C,-1) # B,H_p/w, W_p/w ,w1,w2,c
        x= x.transpose(1,2) # B,L,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = 2 * H * W
        K = 4
        B = B // 2
        
        #self.merge_x(x), =  B//2,C,2*H*W
        #print('forward_core')
        x_wh = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,L,  horizon,vertical
        x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,L,
        xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,L,
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
        out_y = self.selective_scan(
            xs, 
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        '''out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)'''
        assert out_y.dtype == torch.float

        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #B//2,C,2*H*W
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #B,C,L
        y=out_y[:, 0] + inv_y[:, 0] +  wh_y + invwh_y

        y = torch.cat([y[:, :, 0::2], y[:, :, 1::2]], 0).view(2*B, -1, H*W) #B,C,L, - >2B,C,L
        y= y.reshape(B*2, C, H//self.window_size[0], W//self.window_size[1], self.window_size[0], self.window_size[1]).permute(0, 1, 2, 4, 3, 5).contiguous()
        y=y.view(B*2,C,H,W).permute(0,2,3,1)
        
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out #B,H,W,C




# class SS2D_hilbert_3d_shift2(nn.Module):


class SS2D_continuousmamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            window_size=[8,8],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.window_size=window_size
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn
        #self.selective_scan = selective_scan_ref

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x[:, :, 1::2, :] = torch.flip(x[:, :, 1::2, :],dims=[-1])
        #x=x.view(B,C,-1) # B,H_p/w, W_p/w ,w1,w2,c
        #x=torch.cat((x[:,:,0::2],x[:,:,1::2]),2)
        x=x.view(B,C,-1) # B,H_p/w, W_p/w ,w1,w2,c
        #x=torch.cat((x[:,:,0::2],x[:,:,1::2]),2)
        x= x.transpose(1,2) # B,L,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        #x= x.transpose(1,2) # B,L,C 
        #x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.

        return x.transpose(1, 2).contiguous() # B//2,C,2L

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = 2 * H * W
        K = 4
        B = B // 2
        
        #self.merge_x(x), =  B//2,C,2*H*W
        #print('forward_core')
        x_wh = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,L,  horizon,vertical
        x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,L,
        xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,L,
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
        out_y = self.selective_scan(
            xs, 
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        '''out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)'''
        assert out_y.dtype == torch.float

        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #B//2,C,2*H*W
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y=out_y[:, 0] + inv_y[:, 0] +  wh_y + invwh_y

        #y = torch.cat([y[:, :, 0::2], y[:, :, 1::2]], 0).view(2*B, C, H*W).permute(0,2,1)#2*B, H*W ,C
        y = torch.cat([y[:, :, 0::2], y[:, :, 1::2]], 0)
        y=y.view(2*B,C,H,W)
        y[:, :, 1::2, :] = torch.flip(y[:, :, 1::2, :],dims=[-1]) #B,C,H,W
        y=y.permute(0,2,3,1)
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out #B,H,W,C

class SS2D_bidirection(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            window_size=[8,8],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        print('SS2D_bidirection')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.window_size=window_size
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)

        self.selective_scan = selective_scan_fn
        #self.selective_scan = selective_scan_ref

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = 2 * H * W
        K = 2
        B = B // 2
        
        #self.merge_x(x), =  B//2,C,2*H*W
        #print('forward_core')
        #x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous()
        #x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,L,  horizon,vertical

        x =self.merge_x(x) #B,C,2L
 
        x_reverse =  torch.flip(x, dims=[-1])# sequence reverse  # B//2,2,C,L,
        xs = torch.stack([x,x_reverse], dim=1) # reverse# B//2,2,C,2L,
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
        out_y = self.selective_scan(
            xs, 
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        '''out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)'''



        assert out_y.dtype == torch.float

        #wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #B//2,C,2*H*W
        inv_y = torch.flip(out_y[:, 1:], dims=[-1]).view(B, -1, L)
        #invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y= inv_y+out_y[:, 0]
        y= torch.cat((y[...,0::2],y[...,1::2]),0)
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # print(x.shape)
        y = self.forward_core(x)
    
        # print(y.shape)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B//2, H*W, 2, int(self.expand*C))
        y = torch.cat([y[:, :, 0], y[:, :, 1]], 0).view(B, H, W, int(self.expand*C))#.view(B//2, 2*H, 2*W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out




###################################################################################################################################################
class SS2D_zorder(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        print('SS2D_zorder')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn
        #self.selective_scan = selective_scan_ref

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape

        z_ordered_tensor, original_shape, z_ordered_indices = z_ordering_4d(x)
        z_ordered_tensor_wh, _, _ = z_ordering_4d(torch.transpose(x, dim0=2, dim1=3).contiguous()) 
        #print("Z-ordered tensor:", z_ordered_tensor)
        x=z_ordered_tensor.view(B,C,H,W).contiguous()
        x_wh=z_ordered_tensor_wh.view(B,C,W,H).contiguous() 

        L = 2 * H * W
        K = 4
        B = B // 2
        
        #self.merge_x(x), =  B//2,C,2*H*W
        #print('forward_core')
        #x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous()
        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,L,  horizon,vertical
        x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,L,
        xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,L,
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
        out_y = self.selective_scan(
            xs, 
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        '''out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)'''



        assert out_y.dtype == torch.float

        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #B//2,C,2*H*W
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        y=  out_y[:, 0]+ inv_y[:, 0]+wh_y+invwh_y
        
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H*W, 2, int(C))
        y = torch.cat([y[:, :, 0], y[:, :, 1]], 0).view(B*2, H, W, int(C))#.view(B//2, 2*H, 2*W, -1)

        y= y.permute(0,3,1,2).view(B*2,C,-1)
        restored_tensor = restore_z_ordered_tensor_4d(y,(B*2, int(C),H,W), z_ordered_indices)
        restored_tensor=restored_tensor.permute(0,2,3,1).contiguous()

        return restored_tensor

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # print(x.shape)
        y = self.forward_core(x)
        assert y.dtype == torch.float32
        # print(y.shape)

        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out




# class SS2D_hilbert_3d_shift2(nn.Module):
#     def __init__(
#             self,
#             d_model,
#             d_state=16,
#             d_conv=3,
#             expand=2.,
#             dt_rank="auto",
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             dropout=0.,
#             conv_bias=True,
#             bias=False,
#             device=None,
#             dtype=None,
#             shift_size=0,
#             **kwargs,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         print('SS2D_hilbert_3d_shift2')
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

#         self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
#         self.conv2d = nn.Conv2d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             groups=self.d_inner,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#             **factory_kwargs,
#         )
#         self.act = nn.SiLU()
#         k=8
#         self.x_proj = (
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#         )
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
#         del self.x_proj

#         self.dt_projs = (
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#         )
#         self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
#         self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
#         del self.dt_projs

#         self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=k, merge=True)
#         self.Ds = self.D_init(self.d_inner, copies=k, merge=True)

#         self.selective_scan = selective_scan_fn
#         #self.selective_scan = selective_scan_mask_ref
#         #self.selective_scan = selective_scan_mask_fn

#         self.out_norm = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else None
#         self.delta_softplus=True
        
#         p = int(np.log2(8))  # 차수 설정 (예: H와 W, C가 2^p 크기일 때)
#         n = 3  # 3차원 (H, W, C)
#         hilbert_curve = HilbertCurve(p, n)
#         self.coords=  [hilbert_curve.point_from_distance(i) for i in range(2 ** (p * n))]

#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
#                 **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError

#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)

#         dt_proj.bias._no_reinit = True

#         return dt_proj

#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)
#         if copies > 1:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log

#     @staticmethod
#     def D_init(d_inner, copies=1, device=None, merge=True):
#         D = torch.ones(d_inner, device=device)
#         if copies > 1:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)
#         D._no_weight_decay = True
#         return D
    
#     def merge_x(self, x): 
#         B, C, H, W = x.shape
#         x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
#         x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
#         return x.transpose(1, 2).contiguous() # B//2,C,2L
#     def unmerge_x(self, x ): 
#         B, C,L = x.shape  #B,C,2L , L=H*W 
#        # x = x.view(B,C,2,-1) # B,,C,2,L
#        # x= torch.cat( (x[:,:,0], x[:,:,1]),0).view(B*2,C,L//2).contiguous() # B,H,W,C


#         odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
#         even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

#         # 홀수, 짝수 순서로 배치
#         x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

#         #print((x-merged_tensor).max())
#         return x

#     def forward_core(self, x: torch.Tensor,shift_mask=None):
#         B, C, H, W = x.shape # BNW,C,W0,W1
#         #print(shift_mask.shape if shift_mask != None else 'None')  # B//2 NW ,1 ,L
#         sc=x
#         #x=x.permute(0,2,3,1) # 

#         x0 = torch.stack((sc[:B//2],sc[B//2:]),1) # B,T,C,H,W
#         x0_wh= (torch.transpose(x0, dim0=-2, dim1=-1).contiguous())

#         x1 = torch.stack((sc[B//2:],sc[:B//2]),1) # B,T,C,H,W
#         x1_wh= (torch.transpose(x1, dim0=-2, dim1=-1).contiguous())

#         #x_wh= x = torch.stack((sc[B//2:],sc[:B//2]),1) # B,T,C,H,W
#         h0_ordered_tensor = apply_hilbert_curve_3d_vectorized(x0,self.coords)#b,c.2L
#         h0_ordered_tensor_wh = apply_hilbert_curve_3d_vectorized(x0_wh,self.coords)#b,c.2L
        
#         h1_ordered_tensor_wh = apply_hilbert_curve_3d_vectorized(x1_wh,self.coords)#b,c.2L
#         h1_ordered_tensor = apply_hilbert_curve_3d_vectorized(x1,self.coords)#b,c.2L
        
#         #print("Z-ordered tensor:", z_ordered_tensor)

#         L = 2 * H * W #inter 
#         K = 8 #4방향 
        
#         if shift_mask != None:
#             sm=shift_mask
#             shift_mask0=torch.stack((sm[:B//2],sm[:B//2]),1).view(B//2,2,1,H,W)
#             shift_mask0_wh= torch.transpose(shift_mask0, dim0=-2, dim1=-1).contiguous()
#             # shift_mask1=torch.stack((sm[B//2:],sm[B//2:]),1).view(B//2,2,1,H,W)
#             # shift_mask1_wh= torch.transpose(shift_mask1, dim0=-2, dim1=-1).contiguous()
#             #shift_mask=shift_mask[:B//2].view(B//2,1,H,W)
#             #print(shift_mask.shape)
#             h0_ordered_mask = apply_hilbert_curve_3d_vectorized(shift_mask0,self.coords) #B,c,2l
#             h0_ordered_mask_wh = apply_hilbert_curve_3d_vectorized(shift_mask0_wh,self.coords) 
#             # h1_ordered_mask = apply_hilbert_curve_3d_vectorized(shift_mask1,self.coords) #B,c,2l
#             # h1_ordered_mask_wh = apply_hilbert_curve_3d_vectorized(shift_mask1_wh,self.coords) 
#             # z_ordered_mask=z_ordered_mask.view(B,1,H,W).contiguous()
#             # z_ordered_mask_wh=z_ordered_mask_wh.view(B,1,W,H).contiguous() 
#             m0_hwwh = torch.stack([h0_ordered_mask, h0_ordered_mask_wh ], dim=1) # B//2,2,C,2L,  horizon,vertical
#             m0_hwwh_reverse =  torch.flip(m0_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
            
#             shift_mask0 = torch.cat([m0_hwwh,m0_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
#             #shift_mask = ms.float().view(B//2, -1, L).unsqueeze(-2).to(x.device) # B//2 , K , 1, L
#             shift_mask0 = shift_mask0.to(x.device) # B//2 , K , 1, L
            
#             # m1_hwwh = torch.stack([h1_ordered_mask, h1_ordered_mask_wh ], dim=1) # B//2,2,C,2L,  horizon,vertical
#             # m1_hwwh_reverse =  torch.flip(m1_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
            
#             # shift_mask1 = torch.cat([m1_hwwh,m1_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
#             # #shift_mask = ms.float().view(B//2, -1, L).unsqueeze(-2).to(x.device) # B//2 , K , 1, L
#             # shift_mask1 = shift_mask1.to(x.device) # B//2 , K , 1, L

#             shift_mask=torch.cat((shift_mask0,shift_mask0),1)# B//2 , 8 , 1, L
#             #print(shift_mask.shape)
            

#         B = B // 2

#         #self.merge_x(x), =  B//2,C,2*H*W
#         #print('forward_core')
#         #x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous() 

       
#         x_hwwh = torch.stack([h0_ordered_tensor, h0_ordered_tensor_wh,h1_ordered_tensor, h1_ordered_tensor_wh], dim=1) #B//2,2,C,2L
#         #.view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
#         x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
#         xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
        
#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ## projection
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#         xs = xs.float().view(B, -1, L) # B, 4 *c , L
#         Bs = Bs.float().view(B, K, -1, L) # B, 4, d_state , L
        
        

#         Cs = Cs.float().view(B, K, -1, L)# B, 4, d_state , L
#         Ds = self.Ds.float().view(-1) 
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
#         #dt_projs_bias = self.dt_projs_bias.float().view(-1)
#         dt_projs_bias = self.dt_projs_bias.float()
#         if dt_projs_bias is not None:
#             #print(dts.shape , dt_projs_bias.shape)
#             dts = dts + dt_projs_bias[...,None].float()
#         if self.delta_softplus is True :
#             dts = F.softplus(dts)
#         if shift_mask != None:
#             dts= dts*shift_mask #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L
#             #Bs = Bs* shift_mask#  B//2, K, d_state , L *  B//2 , K , 1, L , b=0이면 현재입력 무시
#             #Cs= Cs*shift_mask
            
#         dts = dts.contiguous().float().view(B, -1, L) # B, 4 *d , L

#         '''
#         torch.Size([2048])   4* 512
#         torch.Size([2048])
#         torch.Size([4096])
#         torch.Size([4096])
#         '''
#         out_y = self.selective_scan(
#             xs, #u
#             dts,
#             As, 
#             Bs, 
#             Cs, 
#             Ds, 
#             z=None,
#             delta_bias=None,
#             delta_softplus=False,
#             return_last_state=False,
#         ).view(B, K, -1, L)

#         '''out_y = self.selective_scan(
#             xs, dts,
#             As, Bs, Cs, Ds, 
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#         ).view(B, K, -1, L)'''



#         assert out_y.dtype == torch.float
#         #print(out_y.device)

#         '''
#         def unmerge_x(self, x , H,W): 
#         B, C,L = x.shape  #B,C,2L , L=H*W 
#         x=x.transpose(x,1,2).contiguous().view(B,-1,2,C) # B, L,2,C
#         x= torch.cat( (x[:,:,0], x[:,:,1],0).view(B*2,H,W,C)).contiguous() # B,H,W,C

#         return x
#         '''
#         #B==B//2
#         y0= out_y[:, 0]
#         wh_y0= out_y[:, 1] #B//2,C,2L
#         y1= out_y[:, 2]
#         wh_y1= out_y[:, 3] #B//2,C,2L

#         inv_y = torch.flip(out_y[:, 4:], dims=[-1]).view(B, K//2, -1, L)
        
#         inv_y0=inv_y[:,0]
#         inv_y1=inv_y[:,2]
#         invwh_y0 = inv_y[:, 1]#B//2,C,2L
#         invwh_y1 = inv_y[:, 3]#B//2,C,2L
        
#         wh_y = wh_y0+wh_y1+ invwh_y0+invwh_y1 # B//2,C,2L

#         #wh_y= self.unmerge_x(wh_y)# B,C,L
#         wh_y = restore_hilbert_curve_3d_vectorized(wh_y, self.coords , (B,2, C,  W,H)   )#.view(batch, t, channel, height, width)
#         wh_y = torch.transpose(wh_y,-2,-1).contiguous()
#         #.view(B, -1, W, Hs)
        

#         y= y0+y1+ inv_y0+inv_y1
#         #y= self.unmerge_x(y) # B,C,L
#         y = restore_hilbert_curve_3d_vectorized(y,self.coords,  (B,2, C, H, W) )#.view(batch,t, channel, height, width)

#         y= y + wh_y
#         y= y.view(B*2,C,H,W)
#         y= y.permute(0,2,3,1).contiguous()
 
#         return y

#     def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
#         B, H, W, C = x.shape
#         xz = self.in_proj(x)
#         x, z = xz.chunk(2, dim=-1)
#         x = x.permute(0, 3, 1, 2).contiguous()
#         x = self.act(self.conv2d(x))
#         # print(x.shape)
#         y = self.forward_core(x, shift_mask )
#         assert y.dtype == torch.float32
#         # print(y.shape)

#         y = self.out_norm(y)
#         y = y * F.silu(z)
        
#         out = self.out_proj(y)

#         if self.dropout is not None:
#             out = self.dropout(out)
#         return out





# class tTensor(torch.Tensor):
#     @property
#     def shape(self):
#         shape = super().shape
#         return tuple([int(s) for s in shape])
# to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])


# class SS2D_v2_shift(nn.Module):
#     def __init__(
#             self,
#             d_model, #embedd_dim
#             d_conv=3, #default to 3 for 2D
#             conv_init=None,
#             expand=2,
#             headdim=64, #default to 64
#             ngroups=1,
#             A_init_range=(1, 16),
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init_floor=1e-4,
#             dt_limit=(0.0, float("inf")),
#             learnable_init_states=False,
#             activation="silu", #default to silu
#             bias=False,
#             conv_bias=True,
#             # Fused kernel and sharding options
#             chunk_size=32,
#             use_mem_eff_path=False, #default to False, for custom implementation
#             layer_idx=None,  # Absorb kwarg for general module
#             device=None,
#             dtype=None,
#             linear_attn_duality=False,
#             d_state = 16,
#             **kwargs
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         print('SS2D_v2_shift')
#         self.d_model = d_model
#         self.d_conv = d_conv
#         self.conv_init = conv_init
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.headdim = headdim
#         self.d_state = d_state
#         if ngroups == -1:
#             ngroups = self.d_inner // self.headdim #equivalent to multi-head attention
#         self.ngroups = ngroups
#         assert self.d_inner % self.headdim == 0
#         self.nheads = self.d_inner // self.headdim
#         self.dt_limit = dt_limit
#         self.learnable_init_states = learnable_init_states
#         self.activation = activation
#         #convert chunk_size to triton.language.int32
#         self.chunk_size = chunk_size#torch.tensor(chunk_size,dtype=torch.int32)
#         self.use_mem_eff_path = use_mem_eff_path
#         self.layer_idx = layer_idx
#         self.ssd_positve_dA = kwargs.get('ssd_positve_dA', True) #default to False, ablation for linear attn duality
#         # Order: [z, x, B, C, dt]
#         d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
#         self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs) #

#         conv_dim = self.d_inner + 2 * self.ngroups * self.d_state


#         self.conv2d = nn.Conv2d(
#             in_channels=conv_dim,
#             out_channels=conv_dim,
#             groups=conv_dim,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#             **factory_kwargs,
#         )
#         if self.conv_init is not None:
#             nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
#         # self.conv1d.weight._no_weight_decay = True

#         if self.learnable_init_states:
#             self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
#             self.init_states._no_weight_decay = True

#         self.act = nn.SiLU()

#         # Initialize log dt bias
#         dt = torch.exp(
#             torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         )
#         dt = torch.clamp(dt, min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         self.dt_bias = nn.Parameter(inv_dt)
#         # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
#         # name.endswith("bias") in param_grouping.py
#         self.dt_bias._no_weight_decay = True

#         # A parameter
#         assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
#         A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
#         A_log = torch.log(A).to(dtype=dtype)
#         self.A_log = nn.Parameter(A_log)
#         # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
#         self.A_log._no_weight_decay = True

#         # D "skip" parameter
#         self.D = nn.Parameter(torch.ones(self.nheads, device=device))
#         self.D._no_weight_decay = True

#         # modified from RMSNormGated to layer norm
#         #assert RMSNormGated is not None
#         #self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
#         self.norm = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

#         #linear attention duality
#         self.linear_attn_duality = linear_attn_duality
#         self.kwargs = kwargs

#     def non_casual_linear_attn(self, x, dt, A, B, C, D, shift_mask, H=None, W=None):
#         '''
#         non-casual attention duality of mamba v2
#         x: (B, L, H, D), equivalent to V in attention
#         dt: (B, L, nheads)
#         A: (nheads) or (d_inner, d_state)
#         B: (B, L, d_state), equivalent to K in attention
#         C: (B, L, d_state), equivalent to Q in attention
#         D: (nheads), equivalent to the skip connection
#         shift_mask: B,1,L
#         '''

#         batch, seqlen, head, dim = x.shape
#         dstate = B.shape[2]
#         V = x.permute(0, 2, 1, 3) # (B, H, L, D)
#         dt = dt.permute(0, 2, 1) # (B, H, L)
#         dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
#         if shift_mask != None :
#             shift_mask= shift_mask.unsqueeze(-1).to(dA.device) # B,1,L,1
#             dA= dA* shift_mask[:batch]  
#         #if self.ssd_positve_dA: dA = -dA

#         V_scaled = V * dA
#         K = B.view(batch, 1, seqlen, dstate)# (B, 1, L, D)
#         #if shift_mask != None : # B,1,L,1
#             #K= K* shift_mask[:batch]
#         if getattr(self, "__DEBUG__", False):
#             A_mat = dA.cpu().detach().numpy()
#             A_mat = A_mat.reshape(batch, -1, H, W)
#             setattr(self, "__data__", dict(
#                 dA=A_mat, H=H, W=W, V=V,))

#         if self.ngroups == 1:
#             ## get kv via transpose K and V
#             KV = K.transpose(-2, -1) @ V_scaled # (B, H, dstate, D)  ,    (B, 1, d_state, L ) @ (B, H, L, D)
#             Q = C.view(batch, 1, seqlen, dstate)#.repeat(1, head, 1, 1)
#             x = Q @ KV # (B, H, L, D)   , (B, H, L, dstate) @ (B, H, dstate, D) 
#             x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
#             x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
#         else:
#             assert head % self.ngroups == 0
#             dstate = dstate // self.ngroups
#             K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)
#             V_scaled = V_scaled.view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
#             Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)

#             KV = K.transpose(-2, -1) @ V_scaled # (B, H//g, g, dstate, D)
#             x = Q @ KV # (B, H//g, g, L, D)
#             V_skip = (V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
#             x = x + V_skip # (B, H//g, g, L, D)
#             x = x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim) # (B, L, H, D)
#             x = x.contiguous()

#         return x

#     def non_casual_attn(self, x, dt, A, B, C, D, shift_mask, H=None, W=None):
#         '''
#         non-casual attention duality of mamba v2
#         x: (B, L, H, D), equivalent to V in attention
#         dt: (B, L, nheads)
#         A: (nheads) or (d_inner, d_state)
#         B: (B, L, d_state), equivalent to K in attention
#         C: (B, L, d_state), equivalent to Q in attention
#         D: (nheads), equivalent to the skip connection
#         shift_mask: B,1,L
#         '''

#         batch, seqlen, head, dim = x.shape
#         dstate = B.shape[2]
#         V = x.permute(0, 2, 1, 3) # (B, H, L, D)
#         dt = dt.permute(0, 2, 1) # (B, H, L)
#         if shift_mask != None :
#             shift_mask= shift_mask.to(dt.device) # B,1,L,1
#             dt= dt* shift_mask[:batch]  
#         #if self.ssd_positve_dA: dA = -dA
#         dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1) #(batch, 1, seqlen, 1)

#         # if shift_mask is not None:
#         #     shift_mask=shift_mask # NW,N,N
#         #     nw   = shift_mask.shape[0] # mask: nW, N, N
#         #     shift_mask= shift_mask.unsqueeze(1).unsqueeze(0) # 1, nW, 1,N, N
#         #     # attn = attn.view(B // nW, nW, self.num_heads, N, N) + 


#         V_scaled = V * dA
#         K = B.view(batch, 1, seqlen, dstate)# (B, 1, L, d_state))
#         #if shift_mask != None : # B,1,L,1
#             #K= K* shift_mask[:batch]
#         if getattr(self, "__DEBUG__", False):
#             A_mat = dA.cpu().detach().numpy()
#             A_mat = A_mat.reshape(batch, -1, H, W)
#             setattr(self, "__data__", dict(
#                 dA=A_mat, H=H, W=W, V=V,))

#         if self.ngroups == 1:
#             ## get kv via transpose K and V

#             QK= C.view(batch, 1, seqlen, dstate) @  K.transpose(-2, -1) #   (B, 1, L, dstate)    @ B, 1 ,d_state, L)  =  B,1, L,L

#             x = QK @ V #B,1, L,L @ ( # (B, H, L, D) # B,H,L,D
#             x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
#             x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
#         else:
#             assert head % self.ngroups == 0
#             dstate = dstate // self.ngroups
#             K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)
#             V_scaled = V_scaled.view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
#             Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)

#             KV = K.transpose(-2, -1) @ V_scaled # (B, H//g, g, dstate, D)
#             x = Q @ KV # (B, H//g, g, L, D)
#             V_skip = (V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
#             x = x + V_skip # (B, H//g, g, L, D)
#             x = x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim) # (B, L, H, D)
#             x = x.contiguous()

#         return x


#     def forward(self, u,shift_mask):
#         """
#         u: (B,C,H,W)
#         Returns: same shape as u
#         """
#         b,H,W,c= u.shape
#         u= u.view(b,-1,c)
#         b=b//2

#         batch, seqlen, dim = u.shape

#         #z_ordered_tensor, original_shape, z_ordered_indices = z_ordering_4d(x)
#         #z_ordered_tensor_wh, _, _ = z_ordering_4d(torch.transpose(x, dim0=2, dim1=3).contiguous()) 
        


#         zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
#         A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
#         initial_states=repeat(self.init_states, "... -> b ...", b= batch) if self.learnable_init_states else None
#         dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)


#         z, xBC, dt = torch.split(
#             zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
#         )
#         dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
#         assert self.activation in ["silu", "swish"]


#         #2D Convolution
#         xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         xBC = self.act(self.conv2d(xBC))
#         xBC = xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()

#         # Split into 3 main branches: X, B, C
#         # These correspond to V, K, Q respectively in the SSM/attention duality
#         x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
#         x, dt, A, B, C = to_ttensor(x, dt, A, B, C)
#         y0 = self.non_casual_attn(
#             rearrange(x[b:], "b l (h p) -> b l h p", p=self.headdim),
#             dt[b:], A, B[b:], C[:b], self.D,shift_mask, H, W
#         )
        
#         y1 = self.non_casual_attn(
#             rearrange(x[:b], "b l (h p) -> b l h p", p=self.headdim),
#             dt[:b], A, B[:b], C[b:], self.D, shift_mask,H, W
#         )

#         # if self.linear_attn_duality:
#         # else:
#         #     # if self.kwargs.get('bidirection', False):
#         #     #     #assert self.ngroups == 2 #only support bidirectional with 2 groups
#         #     #     x = to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)).chunk(2, dim=-2)
#         #     #     B = to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
#         #     #     C = to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
#         #     #     dt = dt.chunk(2, dim=-1) # (B, L, nheads) -> (B, L, nheads//2)*2
#         #     #     A, D = A.chunk(2, dim=-1), self.D.chunk(2,dim=-1) # (nheads) -> (nheads//2)*2
#         #     #     y_forward = mamba_chunk_scan_combined(
#         #     #         x[0], dt[0], A[0], B[0], C[0], chunk_size=self.chunk_size, D=D[0], z=None, seq_idx=seq_idx,
#         #     #         initial_states=initial_states, **dt_limit_kwargs
#         #     #     )
#         #     #     y_backward = mamba_chunk_scan_combined(
#         #     #         x[1].flip(1), dt[1].flip(1), A[1], B[1].flip(1), C[1].flip(1), chunk_size=self.chunk_size, D=D[1], z=None, seq_idx=seq_idx,
#         #     #         initial_states=initial_states, **dt_limit_kwargs
#         #     #     )
#         #     #     y = torch.cat([y_forward, y_backward.flip(1)], dim=-2)
#         #     # else:

#         #     print('x[:b]',x.shape)
#         #     print('A',A.shape)
#         #     print('B[:b]',B.shape)
#         #     print('C[:b]',C.shape)
#         #     print('D',self.D.shape)
#         #     print('dt[:b]',dt.shape)
#         #     print('initial_states',initial_states)
#         #     y0 = mamba_chunk_scan_combined(
#         #         to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim).contiguous()),
#         #         to_ttensor(dt.contiguous()),
#         #         to_ttensor(A.contiguous()),
#         #         to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups).contiguous()),
#         #         to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups).contiguous()),
#         #         chunk_size=self.chunk_size,
#         #         D=to_ttensor(self.D),
#         #         z=None,
#         #         seq_idx=None,
#         #         initial_states=initial_states,
#         #         **dt_limit_kwargs,
#         #     )
#         #     # y1 = mamba_chunk_scan_combined(
#         #     #     to_ttensor(rearrange(x[b:,...], "b l (h p) -> b l h p", p=self.headdim).contiguous()),
#         #     #     to_ttensor(dt[b:].contiguous()),
#         #     #     to_ttensor(A.contiguous()),
#         #     #     to_ttensor(rearrange(B[b:,...], "b l (g n) -> b l g n", g=self.ngroups).contiguous()),
#         #     #     to_ttensor(rearrange(C[b:,...], "b l (g n) -> b l g n", g=self.ngroups).contiguous()),
#         #     #     chunk_size=self.chunk_size,
#         #     #     D=to_ttensor(self.D),
#         #     #     z=None,
#         #     #     seq_idx=None,
#         #     #     initial_states=initial_states,
#         #     #     **dt_limit_kwargs,
#         #     # )

#         y=torch.cat((y0,y1),0)
#         y = rearrange(y, "b l h p -> b l (h p)")

#         # # Multiply "gate" branch and apply extra normalization layer
#         # y = self.norm(y, z)
#         y = self.norm(y)
#         y = y*z
#         out = self.out_proj(y)
#         out=out.view(batch,H,W,-1)
#         return out
        

# class SS2D_v2(nn.Module):
#     def __init__(
#             self,
#             d_model, #embedd_dim
#             d_conv=3, #default to 3 for 2D
#             conv_init=None,
#             expand=2,
#             headdim=64, #default to 64
#             ngroups=1,
#             A_init_range=(1, 16),
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init_floor=1e-4,
#             dt_limit=(0.0, float("inf")),
#             learnable_init_states=False,
#             activation="silu", #default to silu
#             bias=False,
#             conv_bias=True,
#             # Fused kernel and sharding options
#             chunk_size=32,
#             use_mem_eff_path=False, #default to False, for custom implementation
#             layer_idx=None,  # Absorb kwarg for general module
#             device=None,
#             dtype=None,
#             linear_attn_duality=False,
#             d_state = 16,
#             **kwargs
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         print('SS2D_v2_zorder_shift')
#         self.d_model = d_model
#         self.d_conv = d_conv
#         self.conv_init = conv_init
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.headdim = headdim
#         self.d_state = d_state
#         if ngroups == -1:
#             ngroups = self.d_inner // self.headdim #equivalent to multi-head attention
#         self.ngroups = ngroups
#         assert self.d_inner % self.headdim == 0
#         self.nheads = self.d_inner // self.headdim
#         self.dt_limit = dt_limit
#         self.learnable_init_states = learnable_init_states
#         self.activation = activation
#         #convert chunk_size to triton.language.int32
#         self.chunk_size = chunk_size#torch.tensor(chunk_size,dtype=torch.int32)
#         self.use_mem_eff_path = use_mem_eff_path
#         self.layer_idx = layer_idx
#         self.ssd_positve_dA = kwargs.get('ssd_positve_dA', True) #default to False, ablation for linear attn duality
#         # Order: [z, x, B, C, dt]
#         d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
#         self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs) #

#         conv_dim = self.d_inner + 2 * self.ngroups * self.d_state


#         self.conv2d = nn.Conv2d(
#             in_channels=conv_dim,
#             out_channels=conv_dim,
#             groups=conv_dim,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#             **factory_kwargs,
#         )
#         if self.conv_init is not None:
#             nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
#         # self.conv1d.weight._no_weight_decay = True

#         if self.learnable_init_states:
#             self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
#             self.init_states._no_weight_decay = True

#         self.act = nn.SiLU()

#         # Initialize log dt bias
#         dt = torch.exp(
#             torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         )
#         dt = torch.clamp(dt, min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         self.dt_bias = nn.Parameter(inv_dt)
#         # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
#         # name.endswith("bias") in param_grouping.py
#         self.dt_bias._no_weight_decay = True

#         # A parameter
#         assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
#         A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
#         A_log = torch.log(A).to(dtype=dtype)
#         self.A_log = nn.Parameter(A_log)
#         # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
#         self.A_log._no_weight_decay = True

#         # D "skip" parameter
#         self.D = nn.Parameter(torch.ones(self.nheads, device=device))
#         self.D._no_weight_decay = True

#         # modified from RMSNormGated to layer norm
#         #assert RMSNormGated is not None
#         #self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
#         self.norm = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

#         #linear attention duality
#         self.linear_attn_duality = linear_attn_duality
#         self.kwargs = kwargs

#     def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
#         '''
#         non-casual attention duality of mamba v2
#         x: (B, L, H, D), equivalent to V in attention
#         dt: (B, L, nheads)
#         A: (nheads) or (d_inner, d_state)
#         B: (B, L, d_state), equivalent to K in attention
#         C: (B, L, d_state), equivalent to Q in attention
#         D: (nheads), equivalent to the skip connection
#         '''

#         batch, seqlen, head, dim = x.shape
#         dstate = B.shape[2]
#         V = x.permute(0, 2, 1, 3) # (B, H, L, D)
#         dt = dt.permute(0, 2, 1) # (B, H, L)
#         dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
#         #if self.ssd_positve_dA: dA = -dA

#         V_scaled = V * dA
#         K = B.view(batch, 1, seqlen, dstate)# (B, 1, L, D)
#         if getattr(self, "__DEBUG__", False):
#             A_mat = dA.cpu().detach().numpy()
#             A_mat = A_mat.reshape(batch, -1, H, W)
#             setattr(self, "__data__", dict(
#                 dA=A_mat, H=H, W=W, V=V,))

#         if self.ngroups == 1:
#             ## get kv via transpose K and V
#             KV = K.transpose(-2, -1) @ V_scaled # (B, H, dstate, D)  ,    (B, 1, d_state, L ) @ (B, H, L, D)
#             Q = C.view(batch, 1, seqlen, dstate)#.repeat(1, head, 1, 1)
#             x = Q @ KV # (B, H, L, D)   , (B, H, L, dstate) @ (B, H, dstate, D) 
#             x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
#             x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
#         else:
#             assert head % self.ngroups == 0
#             dstate = dstate // self.ngroups
#             K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)
#             V_scaled = V_scaled.view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
#             Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)

#             KV = K.transpose(-2, -1) @ V_scaled # (B, H//g, g, dstate, D)
#             x = Q @ KV # (B, H//g, g, L, D)
#             V_skip = (V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
#             x = x + V_skip # (B, H//g, g, L, D)
#             x = x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim) # (B, L, H, D)
#             x = x.contiguous()

#         return x


#     def forward(self, u, seq_idx=None):
#         """
#         u: (B,C,H,W)
#         Returns: same shape as u
#         """
#         b,H,W,c= u.shape
#         u= u.view(b,-1,c)
#         b=b//2

#         batch, seqlen, dim = u.shape

#         #z_ordered_tensor, original_shape, z_ordered_indices = z_ordering_4d(x)
#         #z_ordered_tensor_wh, _, _ = z_ordering_4d(torch.transpose(x, dim0=2, dim1=3).contiguous()) 
        


#         zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
#         A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
#         initial_states=repeat(self.init_states, "... -> b ...", b= batch) if self.learnable_init_states else None
#         dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)


#         z, xBC, dt = torch.split(
#             zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
#         )
#         dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
#         assert self.activation in ["silu", "swish"]


#         #2D Convolution
#         xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         xBC = self.act(self.conv2d(xBC))
#         xBC = xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()

#         # Split into 3 main branches: X, B, C
#         # These correspond to V, K, Q respectively in the SSM/attention duality
#         x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
#         x, dt, A, B, C = to_ttensor(x, dt, A, B, C)
#         y0 = self.non_casual_linear_attn(
#             rearrange(x[b:], "b l (h p) -> b l h p", p=self.headdim),
#             dt[b:], A, B[b:], C[:b], self.D, H, W
#         )
        
#         y1 = self.non_casual_linear_attn(
#             rearrange(x[:b], "b l (h p) -> b l h p", p=self.headdim),
#             dt[:b], A, B[:b], C[b:], self.D, H, W
#         )

#         y=torch.cat((y0,y1),0)
#         y = rearrange(y, "b l h p -> b l (h p)")

#         # # Multiply "gate" branch and apply extra normalization layer
#         # y = self.norm(y, z)
#         y = self.norm(y)
#         y = y*z
#         out = self.out_proj(y)
#         out=out.view(batch,H,W,-1)
#         return out
        

# class Permute(nn.Module):
#     def __init__(self, *args):
#         super().__init__()
#         self.args = args

#     def forward(self, x: torch.Tensor):
#         return x.permute(*self.args)

# # mamba2 support ================================
# class SS2D2 (nn.Module):
#     def __init__(
#         self,
#         # basic dims ===========
#         d_model=128,
#         d_state=64, # now with mamba2, dstate should be bigger...
#         d_conv=3, # < 2 means no conv 
#         exapnd=2,
#         dt_rank="auto",
#         dt_min=0.001,
#         dt_max=0.1,
#         dt_init="random",
#         dt_scale=1.0,
#         dt_init_floor=1e-4,
#         #act_layer=nn.GELU,
#         # dwconv ===============
#         dropout=0.0,
#         conv_bias=True,
#         bias=False,
#         device=None,
#         dtype=None,
#         with_initial_state=False,
#         # ======================
#         # dt init ==============
#         #initialize="v2",
#         # ======================
#         #forward_type="m0",
#         # ======================
#         # ======================
#         **kwargs,    
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         d_inner = int(exapnd * d_model)
#         dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
#         assert d_inner % dt_rank == 0
#         self.with_dconv = d_conv > 1
#         Linear = nn.Linear
#         self.channel_first=False
#         self.ln=  nn.LayerNorm(d_inner)
#         # # tags for forward_type ==============================
#         # checkpostfix = SS2Dv2.checkpostfix
#         # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
#         # self.oact, forward_type = checkpostfix("_oact", forward_type)
#         # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
#         # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
#         # self.out_norm, forward_type = SS2Dv2.get_outnorm(forward_type, d_inner, False)

#         # # forward_type debug =======================================
#         # FORWARD_TYPES = dict(
#         #     m0=partial(self.forward_corem0, force_fp32=False, dstate=d_state),
#         # )
#         # self.forward_core = FORWARD_TYPES.get(forward_type, None)
#         k_group = 4

#         self.disable_z=False
#         self.disable_z_act=False
#         self.out_norm = nn.LayerNorm(d_inner)

#         # in proj =======================================
#         d_proj = d_inner if self.disable_z else (d_inner * 2)
#         self.in_proj = Linear(d_model, d_proj, bias=bias)
#         self.act: nn.Module = nn.SiLU()
        
#         # conv =======================================
#         if self.with_dconv:
#             self.conv2d = nn.Sequential(
#                 Permute(0, 3, 1, 2),
#                 nn.Conv2d(
#                     in_channels=d_inner,
#                     out_channels=d_inner,
#                     groups=d_inner,
#                     bias=conv_bias,
#                     kernel_size=d_conv,
#                     padding=(d_conv - 1) // 2,
#                     **factory_kwargs,
#                 ),
#                 Permute(0, 2, 3, 1),
#             ) 
        
#         # x proj ============================
#         self.x_proj = [
#             nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
#             for _ in range(k_group)
#         ]
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
#         del self.x_proj
        
#         self.oact=False
#         # out proj =======================================
#         self.out_act = nn.GELU() if self.oact else nn.Identity()
#         self.out_proj = Linear(d_inner, d_model, bias=bias)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

#         # if initialize in ["v1"]:
#         #     # simple init dt_projs, A_logs, Ds
#         #     self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
#         #     self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
#         #     self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, dt_rank))) # 0.1 is added in 0430
#         # elif initialize in ["v2"]:
#         #     # simple init dt_projs, A_logs, Ds
#         self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
#         self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
#         self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

#         # init state ============================
#         self.initial_state = None
#         if with_initial_state:
#             self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)), requires_grad=False)

#     def merge_x(self, x): 
#         B, C, H, W = x.shape
#         x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
#         x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
#         return x.transpose(1, 2).contiguous() # B//2,C,2L
#     def unmerge_x(self, x ): 
#         B, C,L = x.shape  #B,C,2L , L=H*W 
#        # x = x.view(B,C,2,-1) # B,,C,2,L
#        # x= torch.cat( (x[:,:,0], x[:,:,1]),0).view(B*2,C,L//2).contiguous() # B,H,W,C


#         odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
#         even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

#         # 홀수, 짝수 순서로 배치
#         x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

#         #print((x-merged_tensor).max())
#         return x
    

#     def forward_core(
#         self,
#         x: torch.Tensor=None, 
#         shift_mask=None,
#         # ==============================
#         force_fp32=False, # True: input fp32
#         chunk_size = 32,
#         dstate = 64,        
#         # ==============================
#         selective_scan_backend = None,
#         scan_mode = "cross2d",
#         scan_force_torch = False,
#         # ==============================
#         **kwargs,
#     ):
#         assert scan_mode in ["unidi", "bidi", "cross2d"]
#         assert selective_scan_backend in [None, "triton", "torch"]
#         x_proj_bias = getattr(self, "x_proj_bias", None)
#         to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

#         N = dstate
#         B, H, W, RD = x.shape
#         K, R = self.A_logs.shape
#         K, R, D = self.Ds.shape
#         assert RD == R * D
#         L = H * W
#         KR = K * R

#         # x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous()
#         # x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,L,  horizon,vertical
#         # x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,L,
#         # xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,L,


#         _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

#         initial_state = None
#         if self.initial_state is not None:
#             assert self.initial_state.shape[-1] == dstate
#             initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
#         xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch) # (B, H, W, 4, D)
#         print('xs',xs.shape)
#         #xs torch.Size([32, 64, 4, 256])ploaded
#         x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
#         if x_proj_bias is not None:
#             x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
#         dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
#         xs = xs.contiguous().view(B, L, KR, D)
#         dts = dts.contiguous().view(B, L, KR)
#         Bs = Bs.contiguous().view(B, L, K, N) # D,dstate
#         Cs = Cs.contiguous().view(B, L, K, N)
#         if force_fp32:
#             xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

#         As = -self.A_logs.to(torch.float).exp().view(KR)
#         Ds = self.Ds.to(torch.float).view(KR, D)
#         dt_bias = self.dt_projs_bias.view(KR)

#         if force_fp32:
#             xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

#         ys, final_state = selective_scan_chunk_fn(
#             xs, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=dt_bias, 
#             initial_states=initial_state, dt_softplus=True, return_final_states=True,
#             backend=selective_scan_backend,
#         )

#         print('ys',ys.shape)
#         y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch)

#         print('y',y.shape)
#         '''
#         torch.Size([512, 64, 256]) MB uploaded
#         torch.Size([512, 64, 256])
#         torch.Size([512, 64, 256])
#         torch.Size([128, 64, 512])
#         torch.Size([128, 64, 512])

#         ys torch.Size([512, 64, 32, 32])loaded
#         y torch.Size([512, 64, 256])B uploaded

#         '''
#         if getattr(self, "__DEBUG__", False):
#             setattr(self, "__data__", dict(
#                 A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=self.Ds,
#                 us=xs, dts=dts, delta_bias=self.dt_projs_bias, 
#                 initial_state=self.initial_state, final_satte=final_state,
#                 ys=ys, y=y, H=H, W=W,
#             ))
#         if self.initial_state is not None:
#             self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

#         y = self.out_norm(y.view(B, H, W, -1))

#         return y.to(x.dtype)
    
#     def forward(self, x, shfit_mask):
#         #x: B,H,W,C

#         x = self.in_proj(x)
#         #if not self.disable_z:
#         x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
#         #    if not self.disable_z_act:
#         #z = self.act(z)
#         z = F.silu(z)
        
#         if self.with_dconv:
#             x = self.conv2d(x) # (b, d, h, w)
#         x = self.act(x) #silu
#         y = self.forward_core(x)
#         y = self.out_act(y)
#         #if not self.disable_z:
#         y = y * z
#         y= self.ln(y)
#         out = self.dropout(self.out_proj(y))
#         return out

# # mamba2 support ================================
# class SS2D2_shiftmask (nn.Module):
#     def __init__(
#         self,
#         # basic dims ===========
#         d_model=128,
#         d_state=64, # now with mamba2, dstate should be bigger...
#         d_conv=3, # < 2 means no conv 
#         exapnd=2,
#         dt_rank="auto",
#         dt_min=0.001,
#         dt_max=0.1,
#         dt_init="random",
#         dt_scale=1.0,
#         dt_init_floor=1e-4,
#         #act_layer=nn.GELU,
#         # dwconv ===============
#         dropout=0.0,
#         conv_bias=True,
#         bias=False,
#         device=None,
#         dtype=None,
#         with_initial_state=False,
#         # ======================
#         # dt init ==============
#         #initialize="v2",
#         # ======================
#         #forward_type="m0",
#         # ======================
#         # ======================
#         **kwargs,    
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         d_inner = int(exapnd * d_model)
#         dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank # embed_dim //16
#         assert d_inner % dt_rank == 0
#         self.with_dconv = d_conv > 1
#         Linear = nn.Linear
#         self.channel_first=False
#         self.ln=  nn.LayerNorm(d_inner)
#         # # tags for forward_type ==============================
#         # checkpostfix = SS2Dv2.checkpostfix
#         # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
#         # self.oact, forward_type = checkpostfix("_oact", forward_type)
#         # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
#         # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
#         # self.out_norm, forward_type = SS2Dv2.get_outnorm(forward_type, d_inner, False)

#         # # forward_type debug =======================================
#         # FORWARD_TYPES = dict(
#         #     m0=partial(self.forward_corem0, force_fp32=False, dstate=d_state),
#         # )
#         # self.forward_core = FORWARD_TYPES.get(forward_type, None)
#         k_group = 4
#         self.delta_softplus=True
#         self.disable_z=False
#         self.disable_z_act=False
#         self.out_norm = nn.LayerNorm(d_inner)

#         # in proj =======================================
#         d_proj = d_inner if self.disable_z else (d_inner * 2)
#         self.in_proj = Linear(d_model, d_proj, bias=bias)
#         self.act: nn.Module = nn.SiLU()
        
#         # conv =======================================
#         if self.with_dconv:
#             self.conv2d = nn.Sequential(
#                 Permute(0, 3, 1, 2),
#                 nn.Conv2d(
#                     in_channels=d_inner,
#                     out_channels=d_inner,
#                     groups=d_inner,
#                     bias=conv_bias,
#                     kernel_size=d_conv,
#                     padding=(d_conv - 1) // 2,
#                     **factory_kwargs,
#                 ),
#                 Permute(0, 2, 3, 1),
#             ) 
        
#         # x proj ============================
#         self.x_proj = [
#             nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
#             for _ in range(k_group)
#         ]
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
#         del self.x_proj
        
#         self.oact=False
#         # out proj =======================================
#         self.out_act = nn.GELU() if self.oact else nn.Identity()
#         self.out_proj = Linear(d_inner, d_model, bias=bias)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

#         # if initialize in ["v1"]:
#         #     # simple init dt_projs, A_logs, Ds
#         #     self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
#         #     self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
#         #     self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, dt_rank))) # 0.1 is added in 0430
#         # elif initialize in ["v2"]:
#         #     # simple init dt_projs, A_logs, Ds
#         self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
#         self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
#         self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

#         # init state ============================
#         self.initial_state = None
#         if with_initial_state:
#             self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)), requires_grad=False)

#     def merge_x(self, x): 
#         B, H, W ,C = x.shape
#         x= x.view(B,H*W,C)
#         x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
#         return x.transpose(1, 2).contiguous() # B//2,C,2L
#     def unmerge_x(self, x ): 
#         B, C,L = x.shape  #B,C,2L , L=H*W 
#        # x = x.view(B,C,2,-1) # B,,C,2,L
#        # x= torch.cat( (x[:,:,0], x[:,:,1]),0).view(B*2,C,L//2).contiguous() # B,H,W,C


#         odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
#         even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

#         # 홀수, 짝수 순서로 배치
#         x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

#         #print((x-merged_tensor).max())
#         return x
    

#     def forward_core(
#         self,
#         x: torch.Tensor=None, 
#         shift_mask=None,
#         # ==============================
#         force_fp32=False, # True: input fp32
#         chunk_size = 64,
#         dstate = 64,        
#         # ==============================
#         selective_scan_backend = None,
#         scan_mode = "cross2d",
#         scan_force_torch = False,
#         # ==============================
#         **kwargs,
#     ):
#         assert scan_mode in ["unidi", "bidi", "cross2d"]
#         assert selective_scan_backend in [None, "triton", "torch"]
#         x_proj_bias = getattr(self, "x_proj_bias", None)
#         to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

#         N = dstate
#         B, H, W, RD = x.shape
#         K, R = self.A_logs.shape
#         K, R, D = self.Ds.shape
#         assert RD == R * D
#         L = H * W
#         KR = K * R

#         # print('L',L)
#         # print('R',R)
#         # print('D',D)
#         # print('K',K)
#         '''
#         L 64b: / 0.014 MB of 0.014 MB uploaded
#         R 8
#         D 32
#         K 4'''
#         # x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous()
#         # x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,L,  horizon,vertical
#         # x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,L,
#         # xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,L,


#         _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

#         initial_state = None
#         if self.initial_state is not None:
#             assert self.initial_state.shape[-1] == dstate
#             initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
        
        
        
#         #x:B,h,w,c
#         #xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch) # (B, H, W, 4, D)
#         L=2*L
#         B=B//2
#         x_wh=torch.transpose(x, dim0=1, dim1=2).contiguous()
#         x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
#         x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
#         xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
#         xs=xs.permute(0,3,1,2) # B, 2L , K , d
#         #print('xs',xs.shape)

#         if shift_mask != None:
#             shift_mask=shift_mask.view(B*2,H,W,1)
#             #print(shift_mask.shape)
#             shift_mask =shift_mask
#             shift_mask_wh=torch.transpose(shift_mask, dim0=1, dim1=2).contiguous()
    
#             shift_mask_hwwh = torch.stack([self.merge_x(shift_mask), self.merge_x(shift_mask_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
#             shift_mask_hwwh_reverse =  torch.flip(shift_mask_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
#             shift_mask = torch.cat([shift_mask_hwwh,shift_mask_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
#             shift_mask=shift_mask.permute(0,3,1,2).to(x.device) # B, 2L , K , 1
                
            
        
#         x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
#         if x_proj_bias is not None:
#             x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
#         dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)

#         # if shift_mask != None :
#         #     print('dts',dts.shape, shift_mask.shape)

#         '''
#         dts torch.Size([256, 128, 4, 8]) torch.Size([256, 128, 4, 1])
#         '''
#         xs = xs.contiguous().view(B, L, KR, D)
#         dts = dts.contiguous().view(B, L, KR)
#         Bs = Bs.contiguous().view(B, L, K, N)
#         Cs = Cs.contiguous().view(B, L, K, N)
#         if force_fp32:
#             xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

#         As = -self.A_logs.to(torch.float).exp().view(KR)
#         Ds = self.Ds.to(torch.float).view(KR, D)
#         dt_bias = self.dt_projs_bias.view(KR)

       
#         if dt_bias is not None:
#             #print(dts.shape , dt_projs_bias.shape)
#             dts = dts + dt_bias
#         if self.delta_softplus is True :
#             dts = F.softplus(dts)
#         if shift_mask != None:
        
#             dts= dts.contiguous().view(B, L, K,R)*shift_mask.view(B, L, K,1) #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L
#             dts = dts.contiguous().view(B, L, KR)

#             #Bs = Bs* shift_mask#  B//2, K, d_state , L *  B//2 , K , 1, L , b=0이면 현재입력 무시
#             #Cs= Cs*shift_mask

#         if force_fp32:
#             xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

#         ys, final_state = selective_scan_chunk_fn(
#             xs, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=None, 
#             initial_states=initial_state, dt_softplus=False, return_final_states=True,
#             backend=selective_scan_backend,
#         )#torch.Size([16, 128, 32, 32])
#         #ys=ys.view(B,H,W,K,RD) # 16,128,4,256
#         ys=ys.view(B,L,K,RD) # 16,128,4,256
#         out_y=ys.permute(0,2,3,1).contiguous() #B,K,RD,L
    
#         wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #B//2,C,2*H*W
#         inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
#         invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#         y= out_y[:,0]+ inv_y[:,0] + wh_y + invwh_y

#         y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H*W, 2, RD)
#         y = torch.cat([y[:, :, 0], y[:, :, 1]], 0).view(B*2, H, W, RD)#.view(B//2, 2*H, 2*W, -1)

#         #y=y.permute(0,2,1).view(B*2,H,W,RD)
#          #B//2,Dinter, HW,HW
#         #y torch.Size([512, 64, 256])B uploaded
        
#         #y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch)

#         if getattr(self, "__DEBUG__", False):
#             setattr(self, "__data__", dict(
#                 A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=self.Ds,
#                 us=xs, dts=dts, delta_bias=self.dt_projs_bias, 
#                 initial_state=self.initial_state, final_satte=final_state,
#                 ys=ys, y=y, H=H, W=W,
#             ))
#         if self.initial_state is not None:
#             self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

#         y = self.out_norm(y)
#             #y.view(B, H, W, -1))

#         return y.to(x.dtype)
    
#     def forward(self, x, shfit_mask):
#         #x: B,H,W,C

#         x = self.in_proj(x)
#         #if not self.disable_z:
#         x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
#         #    if not self.disable_z_act:
#         #z = self.act(z)
#         z = F.silu(z)
        
#         if self.with_dconv:
#             x = self.conv2d(x) # (b, d, h, w)
#         x = self.act(x) #silu
#         y = self.forward_core(x,shfit_mask)
#         y = self.out_act(y)
#         #if not self.disable_z:
#         y = y * z
#         y= self.ln(y)
#         out = self.dropout(self.out_proj(y))
#         return out

# # mamba2 support ================================
# class SS2D2_shiftmask_hd (nn.Module):
#     def __init__(
#         self,
#         # basic dims ===========
#         d_model=128,
#         d_state=64, # now with mamba2, dstate should be bigger...
#         d_conv=3, # < 2 means no conv 
#         exapnd=2,
#         dt_rank="auto",
#         dt_min=0.001,
#         dt_max=0.1,
#         dt_init="random",
#         dt_scale=1.0,
#         dt_init_floor=1e-4,
#         #act_layer=nn.GELU,
#         # dwconv ===============
#         dropout=0.0,
#         conv_bias=True,
#         bias=False,
#         device=None,
#         dtype=None,
#         with_initial_state=False,
#         # ======================
#         # dt init ==============
#         #initialize="v2",
#         # ======================
#         #forward_type="m0",
#         # ======================
#         # ======================
#         **kwargs,    
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         d_inner = int(exapnd * d_model)
#         dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank # embed_dim //16
#         assert d_inner % dt_rank == 0
#         self.with_dconv = d_conv > 1
#         Linear = nn.Linear
#         self.channel_first=False
#         self.ln=  nn.LayerNorm(d_inner)
#         self.headdim = d_model//4
#         # # tags for forward_type ==============================
#         # checkpostfix = SS2Dv2.checkpostfix
#         # self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
#         # self.oact, forward_type = checkpostfix("_oact", forward_type)
#         # self.disable_z, forward_type = checkpostfix("_noz", forward_type)
#         # self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
#         # self.out_norm, forward_type = SS2Dv2.get_outnorm(forward_type, d_inner, False)

#         # # forward_type debug =======================================
#         # FORWARD_TYPES = dict(
#         #     m0=partial(self.forward_corem0, force_fp32=False, dstate=d_state),
#         # )
#         # self.forward_core = FORWARD_TYPES.get(forward_type, None)
#         k_group = 4
#         self.delta_softplus=True
#         self.disable_z=False
#         self.disable_z_act=False
#         self.out_norm = nn.LayerNorm(d_inner)

#         # in proj =======================================
#         d_proj = d_inner if self.disable_z else (d_inner * 2)
#         self.in_proj = Linear(d_model, d_proj, bias=bias)
#         self.act: nn.Module = nn.SiLU()
        
#         # conv =======================================
#         if self.with_dconv:
#             self.conv2d = nn.Sequential(
#                 Permute(0, 3, 1, 2),
#                 nn.Conv2d(
#                     in_channels=d_inner,
#                     out_channels=d_inner,
#                     groups=d_inner,
#                     bias=conv_bias,
#                     kernel_size=d_conv,
#                     padding=(d_conv - 1) // 2,
#                     **factory_kwargs,
#                 ),
#                 Permute(0, 2, 3, 1),
#             ) 
        
#         # x proj ============================
#         self.x_proj = [
#             nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
#             for _ in range(k_group)
#         ]
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
#         del self.x_proj
        
#         self.oact=False
#         # out proj =======================================
#         self.out_act = nn.GELU() if self.oact else nn.Identity()
#         self.out_proj = Linear(d_inner, d_model, bias=bias)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

#         # if initialize in ["v1"]:
#         #     # simple init dt_projs, A_logs, Ds
#         #     self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
#         #     self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
#         #     self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, dt_rank))) # 0.1 is added in 0430
#         # elif initialize in ["v2"]:
#         #     # simple init dt_projs, A_logs, Ds
#         self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
#         self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
#         self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

#         # init state ============================
#         self.initial_state = None
#         if with_initial_state:
#             self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)), requires_grad=False)

#     def merge_x(self, x): 
#         B, H, W ,C = x.shape
#         x= x.view(B,H*W,C)
#         x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
#         return x.transpose(1, 2).contiguous() # B//2,C,2L
#     def unmerge_x(self, x ): 
#         B, C,L = x.shape  #B,C,2L , L=H*W 
#        # x = x.view(B,C,2,-1) # B,,C,2,L
#        # x= torch.cat( (x[:,:,0], x[:,:,1]),0).view(B*2,C,L//2).contiguous() # B,H,W,C


#         odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
#         even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

#         # 홀수, 짝수 순서로 배치
#         x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

#         #print((x-merged_tensor).max())
#         return x
    

#     def forward_core(
#         self,
#         x: torch.Tensor=None, 
#         shift_mask=None,
#         # ==============================
#         force_fp32=False, # True: input fp32
#         chunk_size = 64,
#         dstate = 64,        
#         # ==============================
#         selective_scan_backend = None,
#         scan_mode = "cross2d",
#         scan_force_torch = False,
#         # ==============================
#         **kwargs,
#     ):
#         assert scan_mode in ["unidi", "bidi", "cross2d"]
#         assert selective_scan_backend in [None, "triton", "torch"]
#         x_proj_bias = getattr(self, "x_proj_bias", None)
#         to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

#         N = dstate
#         B, H, W, RD = x.shape
#         K, R = self.A_logs.shape
#         K, R, D = self.Ds.shape
#         assert RD == R * D
#         L = H * W
#         KR = K * R

#         # print('L',L)
#         # print('R',R)
#         # print('D',D)
#         # print('K',K)
#         '''
#         L 64b: / 0.014 MB of 0.014 MB uploaded
#         R 8
#         D 32
#         K 4'''
#         # x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous()
#         # x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,L,  horizon,vertical
#         # x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,L,
#         # xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,L,


#         _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

#         initial_state = None
#         if self.initial_state is not None:
#             assert self.initial_state.shape[-1] == dstate
#             initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
        
        
        
#         #x:B,h,w,c
#         #xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch) # (B, H, W, 4, D)
#         L=2*L
#         B=B//2
#         x_wh=torch.transpose(x, dim0=1, dim1=2).contiguous()
#         x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
#         x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
#         xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
#         xs=xs.permute(0,3,1,2) # B, 2L , K , d
#         #print('xs',xs.shape)

#         if shift_mask != None:
#             shift_mask=shift_mask.view(B*2,H,W,1)
#             #print(shift_mask.shape)
#             shift_mask =shift_mask
#             shift_mask_wh=torch.transpose(shift_mask, dim0=1, dim1=2).contiguous()
    
#             shift_mask_hwwh = torch.stack([self.merge_x(shift_mask), self.merge_x(shift_mask_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
#             shift_mask_hwwh_reverse =  torch.flip(shift_mask_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
#             shift_mask = torch.cat([shift_mask_hwwh,shift_mask_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
#             shift_mask=shift_mask.permute(0,3,1,2).to(x.device) # B, 2L , K , 1
                
            
        
#         x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
#         if x_proj_bias is not None:
#             x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
#         dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)

#         # if shift_mask != None :
#         #     print('dts',dts.shape, shift_mask.shape)

#         '''
#         dts torch.Size([256, 128, 4, 8]) torch.Size([256, 128, 4, 1])
#         '''
#         xs = xs.contiguous().view(B, L, KR, D)
#         dts = dts.contiguous().view(B, L, KR)
#         Bs = Bs.contiguous().view(B, L, K, N)
#         Cs = Cs.contiguous().view(B, L, K, N)
#         if force_fp32:
#             xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

#         As = -self.A_logs.to(torch.float).exp().view(KR)
#         Ds = self.Ds.to(torch.float).view(KR, D)
#         dt_bias = self.dt_projs_bias.view(KR)

       
#         if dt_bias is not None:
#             #print(dts.shape , dt_projs_bias.shape)
#             dts = dts + dt_bias
#         if self.delta_softplus is True :
#             dts = F.softplus(dts)
#         if shift_mask != None:
        
#             dts= dts.contiguous().view(B, L, K,R)*shift_mask.view(B, L, K,1) #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L
#             dts = dts.contiguous().view(B, L, KR)

#             #Bs = Bs* shift_mask#  B//2, K, d_state , L *  B//2 , K , 1, L , b=0이면 현재입력 무시
#             #Cs= Cs*shift_mask

#         if force_fp32:
#             xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

#         ys, final_state = selective_scan_chunk_fn(
#             xs, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=None, 
#             initial_states=initial_state, dt_softplus=False, return_final_states=True,
#             backend=selective_scan_backend,
#         )#torch.Size([16, 128, 32, 32])
#         #ys=ys.view(B,H,W,K,RD) # 16,128,4,256
#         ys=ys.view(B,L,K,RD) # 16,128,4,256
#         out_y=ys.permute(0,2,3,1).contiguous() #B,K,RD,L
    
#         wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #B//2,C,2*H*W
#         inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
#         invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#         y= out_y[:,0]+ inv_y[:,0] + wh_y + invwh_y

#         y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H*W, 2, RD)
#         y = torch.cat([y[:, :, 0], y[:, :, 1]], 0).view(B*2, H, W, RD)#.view(B//2, 2*H, 2*W, -1)

#         #y=y.permute(0,2,1).view(B*2,H,W,RD)
#          #B//2,Dinter, HW,HW
#         #y torch.Size([512, 64, 256])B uploaded
        
#         #y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch)

#         if getattr(self, "__DEBUG__", False):
#             setattr(self, "__data__", dict(
#                 A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=self.Ds,
#                 us=xs, dts=dts, delta_bias=self.dt_projs_bias, 
#                 initial_state=self.initial_state, final_satte=final_state,
#                 ys=ys, y=y, H=H, W=W,
#             ))
#         if self.initial_state is not None:
#             self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

#         y = self.out_norm(y)
#             #y.view(B, H, W, -1))

#         return y.to(x.dtype)
    
#     def forward(self, x, shfit_mask):
#         #x: B,H,W,C

#         x = self.in_proj(x)
#         #if not self.disable_z:
#         x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
#         #    if not self.disable_z_act:
#         #z = self.act(z)
#         z = F.silu(z)
        
#         if self.with_dconv:
#             x = self.conv2d(x) # (b, d, h, w)
#         x = self.act(x) #silu
#         y = self.forward_core(x,shfit_mask)
#         y = self.out_act(y)
#         #if not self.disable_z:
#         y = y * z
#         y= self.ln(y)
#         out = self.dropout(self.out_proj(y))
#         return out
    




# ########################################################################################################################################################################################################################
# ########################################################################################################################################################################################################################

class SS2D_hilbert_shift_rot_inv(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            shift_size=0,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        print('SS2D_hilbert_shift_rot_inv')
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs)
        )
        self.k=8
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.k, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.k, merge=True)

        self.selective_scan = selective_scan_fn
        #self.selective_scan = selective_scan_mask_ref
        #self.selective_scan = selective_scan_mask_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.delta_softplus=True

        
        
        p = int(np.log2(8))  # 힐베르트 곡선의 단계 (order)
        n = 2  # 2차원

        H,W=8,8 #window size 

        # 힐베르트 곡선 객체 생성
        hilbert_curve = HilbertCurve(p, n)

        # 힐베르트 곡선의 전체 좌표 계산
        coords = []
        for y in range(H):
            for x in range(W):
                coords.append((x, y))

        # 각 좌표에 대한 힐베르트 인덱스 계산
        hilbert_indices = []
        for coord in coords:
            x, y = coord
            # 힐베르트 곡선의 크기에 맞게 좌표 조정
            hilbert_index = hilbert_curve.distance_from_point([x, y])
            hilbert_indices.append(hilbert_index)

        # 힐베르트 인덱스에 따라 정렬
        hilbert_indices = np.array(hilbert_indices)
        self.sorted_indices = np.argsort(hilbert_indices)
        # 역순서 인덱스 계산
        self.inverse_indices = np.argsort( self.sorted_indices)


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        x= x.view(B,C,H*W).transpose(1,2) # B,H*W,C 
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, 2 * H * W, C) # 프레임 0,1을 L에 합침.
        return x.transpose(1, 2).contiguous() # B//2,C,2L
    def unmerge_x(self, x ): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
       # x = x.view(B,C,2,-1) # B,,C,2,L
       # x= torch.cat( (x[:,:,0], x[:,:,1]),0).view(B*2,C,L//2).contiguous() # B,H,W,C


        odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

        # 홀수, 짝수 순서로 배치
        x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

        #print((x-merged_tensor).max())
        return x

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # BNW,C,W0,W1
        K = self.k #4방향 
        #print(shift_mask.shape if shift_mask != None else 'None')  # B//2 NW ,1 ,L
        L = H * W #inter 

        #x=x.permute(0,2,3,1) # 

        x_0 = x
        x_1 = torch.rot90(x, k=1, dims=(2, 3)) #시계반대방향 
        x_2 = torch.rot90(x, k=2, dims=(2, 3))
        x_3 = torch.rot90(x, k=3, dims=(2, 3))

        x= torch.stack([x_0,x_1,x_2,x_3],1) # B,4,C,H,W
        x=x.view(B,K//2*C,H,W)
        h_x  = apply_hilbert_curve_2d(x,self.sorted_indices)#B,4c,L
        #h_1, z_ordered_indices1 = apply_hilbert_curve(x_1).contiguous()
        #h_2, z_ordered_indices2 = apply_hilbert_curve(x_2).contiguous()
        #h_3, z_ordered_indices3 = apply_hilbert_curve(x_3).contiguous() 


        #z_ordered_tensor_wh, z_ordered_indices_wh = apply_hilbert_curve(torch.transpose(x, dim0=2, dim1=3).contiguous()) 
        
        #print("Z-ordered tensor:", z_ordered_tensor)
        #x=h_x.view(B,-1,H,W).contiguous()
        
        if shift_mask != None:
            shift_mask=shift_mask.view(B,1,H,W).to(x.device)
            #print(shift_mask.shape)
            s_1=torch.rot90(shift_mask, k=1, dims=(-2, -1))
            s_2=torch.rot90(shift_mask, k=2, dims=(-2, -1))
            s_3=torch.rot90(shift_mask, k=3, dims=(-2, -1))
            shift_mask  = torch.cat([shift_mask,s_1,s_2,s_3],1).contiguous() # B,4,H,W
            shift_mask  = apply_hilbert_curve_2d(shift_mask,self.sorted_indices) #B,4,L
            shift_mask_inv= torch.flip(shift_mask,dims=[-1])
            shift_mask=torch.cat((shift_mask,shift_mask_inv),1) #B,K,1,L
            shift_mask=shift_mask.view(B//2,K,1,-1)
        #self.merge_x(x), =  B//2,C,2*H*W
        #print('forward_core')
        #x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous()
        #x_hwwh = torch.stack([self.merge_x(x), self.merge_x(x_wh)], dim=1).view(B, 2, -1, L) # B//2,2,C,2L,  horizon,vertical
        #x_hwwh_reverse =  torch.flip(x_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
        #xs = torch.cat([x_hwwh,x_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
        
        #x=x.permute(0,1,3,2) #B,K,L,C #BKCL

        # 2L로 만드는 과정 

        h_x=h_x.view(B,K//2*C,H,W)
        L = 2 * H * W #inter 
        B = B // 2

        xs=self.merge_x(h_x).view(B,K//2,C,L)# B//2,KC,2L

        xs_inv= torch.flip(xs,dims=[-1])

        xs = torch.cat([xs,xs_inv], dim=1) # reverse# B//2,8K,C,L,

       

        #xs_inv = torch.flip(xs,dims=[-1])
        #xs=torch.cat((xs,xs_inv),1) #B//2,K,C,2L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) ## projection
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # B, 4 *c , L
        Bs = Bs.float().view(B, K, -1, L) # B, 4, d_state , L
        
        Cs = Cs.float().view(B, K, -1, L)# B, 4, d_state , L
        Ds = self.Ds.float().view(-1) 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        #dt_projs_bias = self.dt_projs_bias.float().view(-1)
        dt_projs_bias = self.dt_projs_bias.float()
        
        if dt_projs_bias is not None:
            #print(dts.shape , dt_projs_bias.shape)
            dts = dts + dt_projs_bias[...,None].float()
        if self.delta_softplus is True :
            dts = F.softplus(dts)
        if shift_mask != None:
            dts= dts*shift_mask #e 0이면 이전 스테이트에 유지 # (B, K, -1, L) * B , K , 1, L
            #Bs = Bs* shift_mask#  B//2, K, d_state , L *  B//2 , K , 1, L , b=0이면 현재입력 무시
            #Cs= Cs*shift_mask
            
        dts = dts.contiguous().float().view(B, -1, L) # B, 4 *d , L
        '''
        torch.Size([2048])   4* 512
        torch.Size([2048])
        torch.Size([4096])
        torch.Size([4096])
        '''
        out_y = self.selective_scan(
            xs, #u
            dts,
            As, 
            Bs, 
            Cs, 
            Ds, 
            z=None,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=False,
        ).view(B, K, -1, L)

        '''out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, 
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)'''



        assert out_y.dtype == torch.float

        '''
        def unmerge_x(self, x , H,W): 
        B, C,L = x.shape  #B,C,2L , L=H*W 
        x=x.transpose(x,1,2).contiguous().view(B,-1,2,C) # B, L,2,C
        x= torch.cat( (x[:,:,0], x[:,:,1],0).view(B*2,H,W,C)).contiguous() # B,H,W,C

        return x
        '''
        #B==B//2



        y= out_y[:,:4]
        y_inv = torch.flip(out_y[:,4:8],dims=[-1]) #B,4,C,L
        out_y=y+y_inv
   
        K=K//2
        out_y=out_y.view(B,K*C,L)
        B=B*2
        L=L//2
        out_y=self.unmerge_x(out_y)
        out_y = reverse_hilbert_curve_2d(out_y,self.inverse_indices,H,W).view(B,K,C,H,W)
        #.view(B,4,C,L)#.view(batch,channel, height, width)


        y0= out_y[:, 0]
        y1= out_y[:, 1]
        y2= out_y[:, 2]
        y3= out_y[:, 3]
        
        y1=torch.rot90(y1, k=-1, dims=(-2, -1))
        y2=torch.rot90(y2, k=-2, dims=(-2, -1))
        y3=torch.rot90(y3, k=-3, dims=(-2, -1))
        
        y=y0+y3+y1+y2 #,B,C,H,W
        #y= y.view(B,-1,L)
        # odd_elements = y[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        # even_elements = y[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)
        # y = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()


        # 홀수, 짝수 순서로 배치

        y= y.permute(0,2,3,1).contiguous()
 
        return y

    def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # print(x.shape)
        y = self.forward_core(x, shift_mask )
        assert y.dtype == torch.float32
        # print(y.shape)

        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out

