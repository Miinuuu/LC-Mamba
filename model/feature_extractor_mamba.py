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
from .VSSBlock import * 

class BiMambaBlock(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                norm_layer=nn.LayerNorm,
                d_state=16,
            ))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
    

class MambaFeature(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dims=[16, 32, 64, 128, 256],
                 depths=(2, 2, 2, 2, 2),
                 conv_stages=3,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super(MambaFeature, self).__init__()
        print('MambaFeature')
        self.num_stages = len(embed_dims)

        self.conv_stages = conv_stages

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                
                else:
                    
                    patch_embed = OverlapPatchEmbed(patch_size=3,
                                                    stride=2,
                                                    in_chans=embed_dims[i - 1],
                                                    embed_dim=embed_dims[i])
                    
                    block = BiMambaBlock(embed_dims[i], depths[i])


                setattr(self, f"patch_embed{i}", patch_embed)
            setattr(self, f"block{i}", block)

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

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)

        features = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i}", None)
            block = getattr(self, f"block{i}", None)

            if i > 0:
                x = patch_embed(x)
            
            x = block(x)

            features.append(x)

        return features



###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_cross(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[11, 11],**kwarg):
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])
                         
                    norm = norm_layer(embed_dims[i])
                    setattr(self, f"norm{i + 1}", norm)
                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
    


###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_zorder(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[11, 11],**kwarg):
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_zorder(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_zorder(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])
                         
                    norm = norm_layer(embed_dims[i])
                    setattr(self, f"norm{i + 1}", norm)
                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
    

###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_local(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[11, 11],**kwarg):
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_local(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_local(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
    

###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_zorder_local(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[11, 11],**kwarg):
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_zorder_local(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_zorder_local(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_v2_local(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[11, 11],**kwarg):
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_v2_local(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_v2_local(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
    


###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_zorder_local_cross(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[11, 11],**kwarg):
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_zorder_local_cross(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_zorder_local_cross(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
    

###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_local_cross(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[11, 11],**kwarg):
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_local_cross(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_local_cross(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
    

###########################################################################################################################################################################
###########################################################################################################################################################################

    



class MotionMamba_cross(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[11, 11],**kwarg):
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_cross(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_cross(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,)for j in range(depths[i])])
                         

                    norm = norm_layer(embed_dims[i])
                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features


###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_zorder_local_shift(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[8, 8],**kwarg):
        super().__init__()
        print('MotionMamba_zorder_local_shift')
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_zorder_local_shift(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2, 
                                    )for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_zorder_local_shift(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2,
                                    )for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
###########################################################################################################################################################################
###########################################################################################################################################################################


###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_hilbert_local_shift(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[8, 8],**kwarg):
        super().__init__()
        print('MotionMamba_hilbert_local_shift')
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_hilbert_local_shift(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2, 
                                    )for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_hilbert_local_shift(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2,
                                    )for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
###########################################################################################################################################################################
###########################################################################################################################################################################


###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_hilbert_local_shift2(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[8, 8],**kwarg):
        super().__init__()
        print('MotionMamba_hilbert_local_shift2')
        print('window_sizes',window_sizes)
        self.window_size=window_sizes
        self.depths = depths
        self.num_stages = len(embed_dims)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_hilbert_local_shift2(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    window_size=self.window_size,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2, 
                                    )for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_hilbert_local_shift2(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    window_size=window_sizes,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2,
                                    )for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
###########################################################################################################################################################################
###########################################################################################################################################################################


###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_hilbert_local_shift2_wos(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[8, 8],**kwarg):
        super().__init__()
        print('MotionMamba_hilbert_local_shift2_wos')
        print('window_sizes',window_sizes)
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_hilbert_local_shift2(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    window_size=window_sizes,
                                    shift_size = int(0) , 
                                    )for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_hilbert_local_shift2(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    window_size=window_sizes,
                                    shift_size = int(0),
                                    )for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
###########################################################################################################################################################################
###########################################################################################################################################################################

###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_hilbert_local_shift_rot(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[8, 8],**kwarg):
        super().__init__()
        print('MotionMamba_hilbert_local_shift_rot')
        self.depths = depths
        self.num_stages = len(embed_dims)

        print(depths)
        print(embed_dims)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_hilbert_local_shift_rot(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    window_sizes=window_sizes,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2, 
                                    )for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_hilbert_local_shift_rot(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    window_sizes=window_sizes,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2,
                                    )for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
###########################################################################################################################################################################
###########################################################################################################################################################################

###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_hilbert_local_shift_rot_inv(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[8, 8],**kwarg):
        super().__init__()
        print('MotionMamba_hilbert_local_shift_rot_inv')
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_hilbert_local_shift_rot_inv(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2, 
                                    )for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_hilbert_local_shift_rot_inv(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2,
                                    )for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
###########################################################################################################################################################################
###########################################################################################################################################################################


###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_hilbert_3d_local_shift(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[8, 8],**kwarg):
        super().__init__()
        print('MotionMamba_hilbert_3d_local_shift')
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_hilbert_3d_local_shift(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2, 
                                    )for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_hilbert_3d_local_shift(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2,
                                    )for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
###########################################################################################################################################################################
###########################################################################################################################################################################



###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_hilbert_3d_local_shift_rot(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[8, 8],**kwarg):
        super().__init__()
        print('MotionMamba_hilbert_3d_local_shift_rot')
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_hilbert_3d_local_shift_rot(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    window_size=window_sizes,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2, 
                                    )for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_hilbert_3d_local_shift_rot(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    window_size=window_sizes,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2,
                                    )for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
###########################################################################################################################################################################
###########################################################################################################################################################################



###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_hilbert_3d_local_shift2(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[8, 8],**kwarg):
        super().__init__()
        print('MotionMamba_hilbert_3d_local_shift2')
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_hilbert_3d_local_shift2(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2, 
                                    )for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_hilbert_3d_local_shift2(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2,
                                    )for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
###########################################################################################################################################################################
###########################################################################################################################################################################


###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba2_local_shift(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[8, 8],**kwarg):
        super().__init__()
        print('MotionMamba2_local_shift')
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock2_local_shift(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2, 
                                    )for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock2_local_shift(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2,
                                    )for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
###########################################################################################################################################################################
###########################################################################################################################################################################






###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_v2_local_shift(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[8, 8],**kwarg):
        super().__init__()
        print('MotionMamba_v2_local_shift')
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_v2_local_shift(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2, 
                                    )for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_v2_local_shift(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=16,
                                    shift_size = int(0) if j%2 ==0 else window_sizes[i-self.conv_stages]//2,
                                    )for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features

####################################
###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_localmamba(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[11, 11],**kwarg):
        super().__init__()
        print('MotionMamba_localmamba')
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_localmamba(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    window_sizes=window_sizes,
                                    d_state=16,)for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_localmamba(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    window_sizes=window_sizes,
                                    d_state=16,)for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
###########################################################################################################################################################################
###########################################################################################################################################################################
class MotionMamba_continuousmamba(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[11, 11],**kwarg):
        super().__init__()
        print('MotionMamba_continuousmamba')
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_continuousmamba(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    window_sizes=window_sizes,
                                    d_state=16,)for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_continuousmamba(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    window_sizes=window_sizes,
                                    d_state=16,)for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
class MotionMamba_bidirection(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[11, 11],**kwarg):
        super().__init__()
        print('MotionMamba_bidirection')
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    if i==self.conv_stages:
                        block = nn.ModuleList([VSSBlock_bidirection(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    window_sizes=window_sizes,
                                    d_state=16,)for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([VSSBlock_bidirection(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    window_sizes=window_sizes,
                                    d_state=16,)for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

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

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        appearence_features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
            else:
                x= patch_embed(x)
                for blk in block:
                    x = blk(x)

            appearence_features.append(x)
        return appearence_features
    

########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################

def feature_extractor_mamba(**kargs):
    model = MambaFeature(**kargs)
    return model

def feature_extractor_zorder(**kargs):
    model = MotionMamba_zorder(**kargs)
    return model

def feature_extractor_cross(**kargs):
    model = MotionMamba_cross(**kargs)
    return model

###########################################################################################################################################################################
###########################################################################################################################################################################
