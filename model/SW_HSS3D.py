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
            zigzag=False,
            conv_first=True,
            shift_mode="padding",
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
        scan_index, merge_index = hilbert_cross_scan_indices(window_size[0], window_size[1], zigzag)
        self.register_buffer('scan_index', scan_index.clone(), persistent=False)
        self.register_buffer('merge_index', merge_index.clone(), persistent=False)

        self.shift_size = list(shift_size) if isinstance(shift_size, (tuple, list)) else [shift_size, shift_size]
        self._scan_mask_cache = {} # (B,H,W,device) -> 스캔 순서 dt 마스크
        self.conv_first = conv_first # True: conv 후 패딩, False: 패딩 후 conv
        if shift_mode not in ("padding", "roll"):
            raise ValueError(f"Unknown shift_mode: {shift_mode!r}")
        self.shift_mode = shift_mode
        self.zigzag = zigzag
        self._roll_plan_cache = {}


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
        H, W = x.shape[-2:]
        out_y = self._scan_sequences(self.cross_scan(x), shift_mask)
        return self.cross_merge(out_y, H, W) # B,H,W,C

    def _scan_sequences(self, xs, shift_mask=None):
        """독립적인 (batch, direction, channel, length) 시퀀스에 SSM 적용."""
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

        return out_y

    @torch.inference_mode(False)
    def _roll_scan_plan(self, H, W, device):
        """roll로 만난 경계 조각을 독립 시퀀스로 분리하는 gather 인덱스.

        각 방향의 Hilbert 순서는 유지하고 패딩 토큰만 생략한다. dt=0인
        패딩을 건너뛰므로 실제 토큰의 state는 padding 모드와 같다.
        계획은 CPU에서 크기별 한 번 생성하고 인덱스만 device에 캐시한다.
        추론 중 생성하더라도 이후 backward에 저장할 수 있는 일반 텐서로 만든다.
        """
        key = (H, W, device)
        if key not in self._roll_plan_cache:
            w0, w1 = self.window_size
            sh, sw = (s % ws for s, ws in zip(self.shift_size, self.window_size))
            ph, pw = (-H) % w0, (-W) % w1
            pt, pl = ph // 2, pw // 2
            pb, pr = ph - pt, pw - pl
            Hp, Wp = H + ph, W + pw
            valid = F.pad(torch.ones(H, W, dtype=torch.bool, device='cpu'), (pl, pr, pt, pb))
            valid = torch.roll(valid, shifts=(-sh, -sw), dims=(0, 1))
            # 마지막 윈도우에 모인 위/아래 및 좌/우 경계 조각을 구별한다.
            row_wrap = torch.arange(Hp, device='cpu') >= Hp - sh
            col_wrap = torch.arange(Wp, device='cpu') >= Wp - sw
            labels = 2 * row_wrap[:, None].long() + col_wrap[None, :].long()
            labels = labels.masked_fill(~valid, -1)
            labels = labels.view(Hp // w0, w0, Wp // w1, w1).permute(0, 2, 1, 3).reshape(-1, w0 * w1)
            labels = labels.repeat(1, 2) # 프레임 0/1은 같은 공간 영역에 속한다.
            scan_index, _ = hilbert_cross_scan_indices(w0, w1, self.zigzag)
            labels = labels[:, scan_index] # nW, K, L
            L = 2 * w0 * w1
            grouped = {}
            for window in range(labels.shape[0]):
                for region in range(4):
                    steps = [torch.where(labels[window, k] == region)[0] for k in range(4)]
                    length = steps[0].numel()
                    if length:
                        grouped.setdefault(length, []).append(torch.stack(steps) + window * L)
            groups = tuple(
                (length, torch.stack(indices, dim=1).reshape(4, -1).to(device))
                for length, indices in sorted(grouped.items())
            )
            if len(self._roll_plan_cache) >= 8:
                self._roll_plan_cache.clear()
            self._roll_plan_cache[key] = ((pt, pb, pl, pr), (sh, sw), groups)
        return self._roll_plan_cache[key]

    def forward_core_roll(self, x):
        """원본 (B,C,H,W) 특징을 roll 후 영역별 독립 state로 스캔한다."""
        B, C, H, W = x.shape
        (pt, pb, pl, pr), (sh, sw), groups = self._roll_scan_plan(H, W, x.device)
        if pt or pb or pl or pr:
            x = F.pad(x, (pl, pr, pt, pb))
        Hp, Wp = x.shape[-2:]
        x = torch.roll(x, shifts=(-sh, -sw), dims=(2, 3))
        xs = self.cross_scan(x)
        BnW, K, _, L = xs.shape
        pairs = B // 2
        nW = BnW // pairs
        flat = xs.view(pairs, nW, K, C, L).permute(0, 2, 3, 1, 4).reshape(pairs, K, C, nW * L)
        out = flat.new_zeros(flat.shape, dtype=torch.float32)
        for length, indices in groups:
            index = indices[None, :, None, :].expand(pairs, K, C, -1)
            packed = torch.gather(flat, 3, index)
            count = indices.shape[-1] // length
            packed = packed.view(pairs, K, C, count, length).permute(0, 3, 1, 2, 4).reshape(-1, K, C, length)
            scanned = self._scan_sequences(packed)
            scanned = scanned.view(pairs, count, K, C, length).permute(0, 2, 3, 1, 4).reshape(pairs, K, C, -1)
            out = out.scatter(3, index, scanned)
        out = out.view(pairs, K, C, nW, L).permute(0, 3, 1, 2, 4).reshape(BnW, K, C, L)
        y = self.cross_merge(out, Hp, Wp)
        y = torch.roll(y, shifts=(sh, sw), dims=(1, 2))
        return y[:, pt:pt + H, pl:pl + W]


    def window_pad(self, H, W):
        """
        cyclic roll 없이 shifted window를 구성하는 패딩 (top, bottom), (left, right).
        기존 중앙 패딩 후 -shift만큼 이동했을 때의 윈도우 경계를 유지한다.
        영상 양끝은 별도 윈도우에 놓이고, 모든 실제 특징은 스캔에 참여한다.
        """
        pads = []
        for n, ws, shift in zip((H, W), self.window_size, self.shift_size):
            pad = (-n) % ws
            before = (pad // 2 - shift) % ws
            after = (-n - before) % ws
            pads.append((before, after))
        return pads

    @torch.inference_mode(False)
    def scan_mask(self, B, H, W, device):
        """
        실제 특징은 1, 패딩만 0인 마스크를 스캔 순서로 변환 (B//2*nW, K, 1, L), 필요 없으면 None.
        dt=0이면 이전 state가 유지되어 패딩 입력이 state에 들어가지 않는다.
        패딩 위치의 출력은 forward에서 제거한다.
        추론에서 만든 캐시도 이후 학습에 재사용할 수 있게 일반 텐서로 만든다.
        입력 크기에만 의존하므로 크기별로 한 번만 만들고 캐시
        """
        key = (B, H, W, device)
        if key not in self._scan_mask_cache:
            (pt, pb), (pl, pr) = self.window_pad(H, W)
            mask = None
            if pt or pb or pl or pr:
                mask = F.pad(torch.ones(B, 1, H, W, device=device), (pl, pr, pt, pb)) # B,1,H_p,W_p
            if mask is not None:
                mask = self.cross_scan(mask) # B//2*nW,K,1,L
            if len(self._scan_mask_cache) >= 8:
                self._scan_mask_cache.clear()
            self._scan_mask_cache[key] = mask
        return self._scan_mask_cache[key]

    def forward(self, x: torch.Tensor, pad_token=None, **kwargs):
        """
        x: (B, H, W, C) ln_1 적용 후, 배치 앞/뒤 절반이 프레임 0/1
        pad_token: conv_first=False 일 때 패딩 위치에 넣을 값 (기존 코드의 ln_1(0) = ln_1.bias), None 이면 0
        """
        B, H, W, C = x.shape
        (pt, pb), (pl, pr) = self.window_pad(H, W)
        padded = pt or pb or pl or pr
        if self.conv_first:
            # 원본 위치에서 conv 후, shift가 반영된 패딩으로 윈도우를 구성
            xz = self.in_proj(x)
            x, z = xz.chunk(2, dim=-1)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))
        else:
            # shift가 반영된 패딩 -> in_proj -> conv
            if padded:
                x_pad = (pad_token if pad_token is not None else x.new_zeros(C)).expand(B, H + pt + pb, W + pl + pr, C).clone()
                x_pad[:, pt:pt + H, pl:pl + W] = x
                x = x_pad
            xz = self.in_proj(x)
            x, z = xz.chunk(2, dim=-1)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))
            z = z[:, pt:pt + H, pl:pl + W]

        if self.shift_mode == "roll" and any(self.shift_size):
            if not self.conv_first:
                # Conv 경계 조건은 padding 모드와 동일하게 유지한다.
                x = x[:, :, pt:pt + H, pl:pl + W].contiguous()
            y = self.forward_core_roll(x)
        else:
            if self.conv_first and padded:
                x = F.pad(x, (pl, pr, pt, pb))
            y = self.forward_core(x, self.scan_mask(B, H, W, x.device))
            y = y[:, pt:pt + H, pl:pl + W]
        assert y.dtype == torch.float32

        # LayerNorm 은 토큰별 연산이라 윈도우 복원 후 적용해도 결과 동일
        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)#B,H,W,C

        if self.dropout is not None:
            out = self.dropout(out)
        return out
