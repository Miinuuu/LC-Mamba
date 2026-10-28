import torch
import torch.nn as nn
import math
from einops import  repeat
import torch.nn.functional as F

try:
    from .mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref 
except:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref 
try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from mamba2.ssd_minimal import selective_scan_chunk_fn


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

        # 힐베르트 스캔 인덱스를 GPU 버퍼로 보관 (numpy 인덱싱은 매 호출 CPU->GPU 복사 + 동기화 발생)
        # persistent=False: state_dict 에 포함되지 않으므로 기존 체크포인트와 호환
        scan_index, merge_index = hilbert_cross_scan_indices(window_size[0], window_size[1])
        self.register_buffer('scan_index', scan_index.clone(), persistent=False)
        self.register_buffer('merge_index', merge_index.clone(), persistent=False)


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
    
    def cross_scan(self, x):
        """
        윈도우 분할 + 힐베르트 순서 + 두 프레임 interleave + 4방향(정/역) 스캔을 gather 한 번으로 수행
        x: (B, C, H, W), 배치 앞/뒤 절반이 프레임 0/1, H, W 는 윈도우 크기의 배수
        출력: (B//2 * nW, K, C, L), L = 2 * w0 * w1
        """
        B, C, H, W = x.shape
        w0, w1 = self.window_size
        x = x.view(2, B // 2, C, H // w0, w0, W // w1, w1).permute(1, 3, 5, 2, 0, 4, 6).reshape(-1, C, 2 * w0 * w1)
        BnW, L = x.shape[0], x.shape[-1]
        K = self.scan_index.shape[0]
        return torch.gather(x.unsqueeze(1).expand(BnW, K, C, L), 3,
                            self.scan_index.view(1, K, 1, L).expand(BnW, K, C, L))

    def cross_merge(self, y, H, W):
        """
        cross_scan 의 역과정: 4방향 출력을 원래 위치로 되돌려 합산 후 윈도우를 이미지로 복원
        y: (B//2 * nW, K, C, L)
        출력: (B, H, W, C)
        """
        BnW, K, C, L = y.shape
        w0, w1 = self.window_size
        y = torch.gather(y, 3, self.merge_index.view(1, K, 1, L).expand(BnW, K, C, L))
        y = (y[:, 0] + y[:, 2]) + (y[:, 1] + y[:, 3]) # (hw + 역hw) + (wh + 역wh), 기존 합산 순서 유지
        y = y.view(-1, H // w0, W // w1, C, 2, w0, w1).permute(4, 0, 1, 5, 2, 6, 3)
        return y.reshape(-1, H, W, C)

    def forward_core(self, x: torch.Tensor,shift_mask=None):
        B, C, H, W = x.shape # B,C,H,W (윈도우 크기 배수로 패딩된 전체 feature)

        xs = self.cross_scan(x) # B//2*nW,4,C,2L
        B, K, _, L = xs.shape

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
            xs.contiguous(), #u
            dts.contiguous(),
            As.contiguous(), 
            Bs.contiguous(), 
            Cs.contiguous(), 
            Ds.contiguous(), 
            #z=None,
            delta_bias=None,
            delta_softplus=False,
            #return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        return self.cross_merge(out_y, H, W) # B,H,W,C


    def forward(self, x: torch.Tensor,shift_mask=None, **kwargs):
        """
        x: (B, H, W, C), H, W 는 윈도우 크기의 배수
        shift_mask: LC_Mamba_Block.scan_mask() 의 스캔 순서 dt 마스크 (B//2*nW, K, 1, L) 또는 None
        """
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y = self.forward_core(x, shift_mask ) #b,h,w,c
        assert y.dtype == torch.float32

        # LayerNorm 은 토큰별 연산이라 윈도우 복원 후 적용해도 결과 동일
        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)#B,H,W,C

        if self.dropout is not None:
            out = self.dropout(out)
        return out
