import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_
from .mamba_layer import *
from .LC_Mamba_Block import * 
from .layers import * 

###########################################################################################################################################################################
###########################################################################################################################################################################

class LC_Mamba_LFE_STFE(nn.Module):
    def __init__(self, 
                 in_chans=3, 
                 embed_dims=[32, 64, 128, 256, 512], 
                 d_state=16,
                 expand=2,
                 depths=[2, 2, 2, 2, 2], 
                 window_sizes=[8, 8],
                 window_shift=1,
                 **kwarg):
        super().__init__()
        print('LC_Mamba_LFE_STFE')
        print('window_sizes',window_sizes)
        print('window_shift',window_shift)

        self.depths = depths
        self.num_stages = len(embed_dims)
        self.conv_stages = self.num_stages - len(window_sizes)

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
                        block = nn.ModuleList([LC_Mamba_Block(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=d_state,
                                    expand=expand,
                                    window_size = window_sizes,
                                    shift_size =  0 if window_shift==-1  else  (0 if j%2 ==0 else window_sizes[i-self.conv_stages]//2), 
                                    )for j in range(depths[i])])


                    else :

                        block = nn.ModuleList([LC_Mamba_Block(  
                                    hidden_dim=embed_dims[i],
                                    norm_layer=nn.LayerNorm,
                                    d_state=d_state,
                                    expand=expand,
                                    window_size =window_sizes,
                                    shift_size =  0 if window_shift==-1  else  (0 if j%2 ==0 else window_sizes[i-self.conv_stages]//2), 
                                    )for j in range(depths[i])])
                         

                setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
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