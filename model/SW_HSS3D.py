import torch
import torch.nn as nn
import math
from einops import  repeat
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref 
from .hilbert_2d import * 

class SW_HSS3D(nn.Module):
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
        print('SW_HSS3D')
        print('window_size',window_size)
        print('shift_size',shift_size)
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


        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.delta_softplus=True

        p = int(np.log2(window_size[0])) 
        n = 2  

        H,W=window_size[0],window_size[1]

        hilbert_curve = HilbertCurve(p, n)

        coords = []
        for y in range(H):
            for x in range(W):
                coords.append((x, y))

        hilbert_indices = []
        for coord in coords:
            x, y = coord
            hilbert_index = hilbert_curve.distance_from_point([x, y])
            hilbert_indices.append(hilbert_index)

        hilbert_indices = np.array(hilbert_indices)
        self.sorted_indices = np.argsort(hilbert_indices)
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

        odd_elements = x[:, :, 1::2]  # L 차원에서 홀수 인덱스 (1, 3, 5, ...)
        even_elements = x[:, :, 0::2]  # L 차원에서 짝수 인덱스 (0, 2, 4, ...)

        # 홀수, 짝수 순서로 배치
        x = torch.cat((even_elements, odd_elements), dim=0).view(B*2,C,L//2).contiguous()

        return x

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # BNW,C,W0,W1

        h_ordered_tensor = apply_hilbert_curve_2d(x,self.sorted_indices)
        h_ordered_tensor_wh = apply_hilbert_curve_2d(torch.transpose(x, dim0=2, dim1=3).contiguous(),self.sorted_indices) 
        
        x=h_ordered_tensor.view(B,C,H,W).contiguous()
        x_wh=h_ordered_tensor_wh.view(B,C,W,H).contiguous() 

        L = 2 * H * W #inter 
        K = 4 #4방향 
        
        if shift_mask != None:
            shift_mask=shift_mask.view(B,1,H,W)
            h_ordered_tensor_wh= apply_hilbert_curve_2d(shift_mask,self.sorted_indices)
            h_ordered_mask_wh= apply_hilbert_curve_2d(torch.transpose(shift_mask, dim0=2, dim1=3).contiguous(),self.sorted_indices) 
            h_ordered_tensor_wh=h_ordered_tensor_wh.view(B,1,H,W).contiguous()
            h_ordered_mask_wh=h_ordered_mask_wh.view(B,1,W,H).contiguous() 
            shift_mask_hwwh = torch.stack([self.merge_x(h_ordered_tensor_wh), self.merge_x(h_ordered_mask_wh)], dim=1).view(B//2, 2, 1, L) # B//2,2,C,2L,  horizon,vertical
            shift_mask_hwwh_reverse =  torch.flip(shift_mask_hwwh, dims=[-1])# sequence reverse  # B//2,2,C,2L,
            shift_mask = torch.cat([shift_mask_hwwh,shift_mask_hwwh_reverse], dim=1) # reverse# B//2,4,C,2L,
            shift_mask = shift_mask.float().view(B//2, -1, L).unsqueeze(-2).to(x.device) # B//2 , K , 1, L
            

        B = B // 2


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

        y= out_y[:, 0]
        wh_y= out_y[:, 1] #B//2,C,2L

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        invwh_y = inv_y[:, 1]#B//2,C,2L
        
        wh_y = wh_y+ invwh_y # B//2,C,2L

        wh_y= self.unmerge_x(wh_y)# B,C,L
        wh_y = reverse_hilbert_curve_2d(wh_y, self.inverse_indices , W,H)#.view(batch, channel, height, width)
        wh_y = torch.transpose(wh_y,2,3).contiguous()
        
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

        y = self.forward_core(x, shift_mask ) #b,h,w,c
        assert y.dtype == torch.float32 

        y = self.out_norm(y)
        y= self.window_reverse(y,self.window_size,H,W)#B,H,W,C
        y = y * F.silu(z)
        
        out = self.out_proj(y)#B,H,W,C

        if self.dropout is not None:
            out = self.dropout(out)
        return out
