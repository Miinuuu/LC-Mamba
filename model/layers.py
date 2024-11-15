import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys
import torch.nn.functional as F
#from .convnext import Block

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )
            
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes,kernel_size=3, stride=2,padding=1):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, kernel_size, stride, padding)
        self.conv2 = conv(out_planes, out_planes, 3, 1, padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


    
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )

class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()
        self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
                                  conv(in_planes*2 // (4*4) + in_else, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  

    def forward(self, motion_feature, x, flow): # /16 /8 /4
        motion_feature = self.upsample(motion_feature) #/4 /2 /1
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
        if flow != None:
            if self.scale != 4:
                flow = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
            x = torch.cat((x, flow), 1)

        x = self.conv(torch.cat([motion_feature, x], 1))
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = self.scale // 4, mode="bilinear", align_corners=False)
            flow = x[:, :4] * (self.scale // 4)
        else:
            flow = x[:, :4]
        mask = x[:, 4:5]
        return flow, mask
    

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2,k=3,s=1,p=1,act_layer=nn.PReLU,):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv2d(in_dim, out_dim, k,s,p))
            else:
                layers.append(nn.Conv2d(out_dim, out_dim, k,s,p))
            layers.extend([
                act_layer(out_dim),
            ])
        self.conv = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class CrossScalePatchEmbed(nn.Module):
    def __init__(self, in_dims=[16,32,64], embed_dim=768):
        super().__init__()
        base_dim = in_dims[0]

        layers = []
        for i in range(len(in_dims)):
            for j in range(2 ** i):
                layers.append(nn.Conv2d(in_dims[-1-i], base_dim, 3, 2**(i+1), 1+j, 1+j))
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Conv2d(base_dim * len(layers), embed_dim, 1, 1)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, xs):
        ys = []
        k = 0
        for i in range(len(xs)):
            for _ in range(2 ** i):
                ys.append(self.layers[k](xs[-1-i]))
                k += 1
        x = self.proj(torch.cat(ys,1))
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

    
class DWConv(nn.Module):
    def __init__(self, dim,k=3,s=1,p=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, k, s, p, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.reshape(B, C, -1).transpose(1, 2)

        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0]*window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    nwB, N, C = windows.shape
    windows = windows.view(-1, window_size[0], window_size[1], C)
    B = int(nwB / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def pad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
        img_mask = torch.zeros((1, h+pad_h, w+pad_w, 1))  # 1 H W 1
        h_slices = (
            slice(0, pad_h//2),
            slice(pad_h//2, h+pad_h//2),
            slice(h+pad_h//2, None),
        )
        w_slices = (
            slice(0, pad_w//2),
            slice(pad_w//2, w+pad_w//2),
            slice(w+pad_w//2, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, window_size
        )  # nW, window_size*window_size, 1
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, float(0.0))
        return nn.functional.pad(
            x,
            (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        ), attn_mask
    return x, None


def depad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
        return x[:, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w, :].contiguous()
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., cond_pos = True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        if cond_pos :
            self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.cond_pos=cond_pos
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.cond_pos:
            x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



