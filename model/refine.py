import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_
from .layers import * 
from .warplayer import *
from .convnext import Block
from .layers2 import * 
from .transforms import * 
from .cbam import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Unet(nn.Module):
    def __init__(self, c, out=3):
        super(Unet, self).__init__()
        self.down0 = Conv2(17+c, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, out, 3, 1, 1)
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

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):

       
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow,c0[0], c1[0]), 1))
        s1 = self.down1(torch.cat((s0, c0[1], c1[1]), 1))
        s2 = self.down2(torch.cat((s1, c0[2], c1[2]), 1))
        s3 = self.down3(torch.cat((s2, c0[3], c1[3]), 1))
        x = self.up0(torch.cat((s3, c0[4], c1[4]), 1))
        x = self.up1(torch.cat((x, s2), 1)) 
        x = self.up2(torch.cat((x, s1), 1)) 
        x = self.up3(torch.cat((x, s0), 1)) 
        x = self.conv(x)
        return torch.sigmoid(x)
    
class Unet_srf2(nn.Module): #8c 4c 2c
    def __init__(self, c, out=3):
        super(Unet_srf2, self).__init__()
        
        self.down0 = Conv2(17 + c ,2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.up0 = deconv( 8*c ,2*c)
        self.up1 = deconv( 4*c,c)
        self.conv = nn.Conv2d(c, out, 3, 1, 1)
        
        #self.lastconv = nn.Conv2d(c ,out,3,1,1)
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

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow,c0[0], c1[0]), 1)) # 256-128
        s1 = self.down1(torch.cat((s0, c0[1], c1[1]), 1)) # 128-64
        x = self.up0(torch.cat((s1, c0[2], c1[2]), 1)) # 32 - 64
        x = self.up1(torch.cat((x, s0), 1))  # 64-128 
        x = self.conv(x)

        return torch.sigmoid(x)
    
    
class Unet_srf1(nn.Module): #8c 4c 2c
    def __init__(self, c, out=3):
        super(Unet_srf1, self).__init__()
        
        self.down0 = Conv2(17 + c ,2*c)
        self.up0 = deconv( 4*c ,2*c)
        self.conv = nn.Conv2d(2*c, out, 3, 1, 1)
        
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

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow,c0[0], c1[0]), 1)) # 256-128
        x = self.up0(torch.cat((s0, c0[1], c1[1]), 1)) # 128-64
        x = self.conv(x)

        return torch.sigmoid(x)
    

class dec6(nn.Module):
    def __init__(self, c, out=3):
        super(dec6, self).__init__()
    
        self.up0 = nn.Sequential(
                         Conv2(17+ 4*c,  4*c,3,1,1)
                        ,Conv2( 4*c,  4*c,3,1,1)
                        ,Conv2( 4*c,  4*c,3,1,1)
                        ,deconv(4*c, 4*c)
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(17+ 2*c+4*c,2*c,3,1,1),
                        Conv2(2*c,2*c,3,1,1),
                        Conv2(2*c,2*c,3,1,1),
                        deconv(2*c, 2*c))
                 
        self.up2 = nn.Sequential(
                        Conv2(17+ 2*c + c, c,3,1,1),
                        Conv2(c, c,3,1,1),
                        Conv2(c, c,3,1,1),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)

class dec6_(nn.Module):
    def __init__(self, c, out=3):
        super(dec6_, self).__init__()
    
        self.up0 = nn.Sequential(
                         Conv2(17+ 4*c,  4*c,3,1,1)
                        ,Conv2( 4*c,  4*c,3,1,1)
                        ,Conv2( 4*c,  4*c,3,1,1)
                        ,deconv(4*c, 2*c)
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(17+ 2*c+2*c,2*c,3,1,1),
                        Conv2(2*c,2*c,3,1,1),
                        Conv2(2*c,2*c,3,1,1),
                        deconv(2*c, c))
                 
        self.up2 = nn.Sequential(
                        Conv2(17+ c + c, c,3,1,1),
                        Conv2(c, c,3,1,1),
                        Conv2(c, c,3,1,1),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)

class dec6dw(nn.Module):
    def __init__(self, c, out=3):
        super(dec6dw, self).__init__()
    
        self.up0 = nn.Sequential(
                         Conv2(17+ 4*c,  4*c,3,1,1)
                        ,DWConv2( 4*c,  4*c,7,1,3, 4*c)
                        ,DWConv2( 4*c,  4*c,7,1,3, 4*c)
                        ,deconv(4*c, 4*c)
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(17+ 2*c+4*c,2*c,3,1,1),
                        DWConv2(2*c,2*c,7,1,3, 2*c),
                        DWConv2(2*c,2*c,7,1,3, 2*c),
                        deconv(2*c, 2*c))
                 
        self.up2 = nn.Sequential(
                        Conv2(17+ 2*c + c, c,3,1,1),
                        DWConv2(c, c,7,1,3, c),
                        DWConv2(c, c,7,1,3, c),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)

class dec4(nn.Module):
    def __init__(self, c, out=3):
        super(dec4, self).__init__()
        
        self.up0 = nn.Sequential(
                         Conv2(17+ 4*c,  4*c,3,1,1)
                        ,Conv2( 4*c,  4*c,3,1,1)
                        ,deconv(4*c, 4*c)
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(17+ 2*c+4*c,2*c,3,1,1),
                        Conv2(2*c,2*c,3,1,1),
                        deconv(2*c, 2*c))
                 
        self.up2 = nn.Sequential(
                        Conv2(17+ 2*c + c, c,3,1,1),
                        Conv2(c, c,3,1,1),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)
    
class dec2(nn.Module):
    def __init__(self, c, out=3):
        super(dec2, self).__init__()
        
        self.up0 = nn.Sequential(
                         Conv2(17+ 4*c,  4*c,3,1,1)
                        ,deconv(4*c, 4*c)
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(17+ 2*c+4*c,2*c,3,1,1),
                        deconv(2*c, 2*c))
                 
        self.up2 = nn.Sequential(
                        Conv2(17+ 2*c + c, c,3,1,1),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)
    
    
class Unet_ldsrf(nn.Module): #8c 4c 2c
    def __init__(self, c, out=3):
        super(Unet_ldsrf, self).__init__()
        
        self.down0 = Conv2(17 + c ,2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.up0 = deconv( 8*c ,2*c)
        self.up1 = deconv( 4*c,c)
        self.conv = nn.Conv2d(c, out, 3, 1, 1)
        
        #self.lastconv = nn.Conv2d(c ,out,3,1,1)
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        
        s0 = self.down0(torch.cat((imgs[0], wimgs[0], masks[0], flows[0],Cs[0]), 1)) # 256-128
        s1 = self.down1(torch.cat((s0, Cs[1]), 1)) # 128-64
        x = self.up0(torch.cat((s1, Cs[2]), 1)) # 32 - 64
        x = self.up1(torch.cat((x, s0), 1))  # 64-128 
        x = self.conv(x)

        return torch.sigmoid(x)
    
    

    
class dec_convnext(nn.Module):
    def __init__(self, c, out=3):
        super(dec_convnext, self).__init__()
    
        self.up0 = nn.Sequential(
                         Conv2(17+4*c,4*c,3,1,1)
                        ,Block( 4*c)
                        ,Block( 4*c)
                        ,deconv(4*c, 4*c)
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(17+2*c+4*c,2*c,3,1,1),
                        Block(2*c),
                        Block(2*c),
                        deconv(2*c, 2*c))
                 
        self.up2 = nn.Sequential(
                        Conv2(17+ 2*c + c, c,3,1,1),
                        Block(c),
                        Block(c),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)

    
    
class dec_convnext_(nn.Module):
    def __init__(self, c, out=3):
        super(dec_convnext_, self).__init__()
    
        self.up0 = nn.Sequential(
                         Conv2(17+4*c,4*c,3,1,1)
                        ,Block( 4*c)
                        ,Block( 4*c)
                        ,deconv(4*c, 2*c)
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(17+2*c+2*c,2*c,3,1,1),
                        Block(2*c),
                        Block(2*c),
                        deconv(2*c, c))
                 
        self.up2 = nn.Sequential(
                        Conv2(17+ c + c, c,3,1,1),
                        Block(c),
                        Block(c),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)

    
class dec_opt(nn.Module):
    def __init__(self, c, out=3):
        super(dec_opt, self).__init__()
    
        self.up0 = nn.Sequential(
                         Conv2(17+4*c,4*c,3,1,1),
                         Conv2(4*c,4*c,3,1,1)
                        #,deconv(4*c, 2*c)
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(17+4*c+2*c,2*c,3,1,1),
                        Conv2(2*c,2*c,3,1,1),
                        #deconv(2*c, c))
        )
        self.up2 = nn.Sequential(
                        Conv2(17+ 2*c+c, c,3,1,1),
                        Conv2(c, c,3,1,1),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x=F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x=F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)

    
class dec_opt_(nn.Module):
    def __init__(self, c, out=3):
        super(dec_opt_, self).__init__()
    
        self.up0 = nn.Sequential(
                         Conv2(17+4*c,4*c,3,1,1),
                        #,deconv(4*c, 2*c)
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(17+4*c+2*c,2*c,3,1,1),
                        #deconv(2*c, c))
        )
        self.up2 = nn.Sequential(
                        Conv2(17+ 2*c+c, c,3,1,1),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x=F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x=F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)
    

            

class dec_dq(nn.Module):
    def __init__(self, c, out=3):
        super(dec_dq, self).__init__()
        print("dec_dq")
    
        self.up0 = nn.Sequential(
                         mlp(17+4*c,4*c*4, 4*c),
                         RB(4*c,4*c),
                         RB(4*c,4*c),
                         deconv(4*c, 2*c)
        )
        
        self.up1 =  nn.Sequential(
                        mlp(17+2*c+2*c,2*c*4, 2*c),
                        RB(2*c,2*c),
                        RB(2*c,2*c),
                        deconv(2*c, c)
        )
        self.up2 = nn.Sequential(
                        mlp(17+c+c,c*4,c),
                        RB(c,c),
                        RB(c,c),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)
            

class dec_dq_bi(nn.Module):
    def __init__(self, c, out=3):
        super(dec_dq_bi, self).__init__()
        print("dec_dq_bi")
    
        self.up0 = nn.Sequential(
                         mlp(17+4*c,4*c*4, 4*c),
                         RB(4*c,4*c),
                         RB(4*c,4*c),
                         #deconv(4*c, 2*c)
        )
        self.up0_= conv(4*c,2*c)
        
        self.up1 =  nn.Sequential(
                        mlp(17+2*c+2*c,2*c*4, 2*c),
                        RB(2*c,2*c),
                        RB(2*c,2*c),
                        #deconv(2*c, c)
        )
        self.up1_= conv(2*c,c)

        self.up2 = nn.Sequential(
                        mlp(17+c+c,c*4,c),
                        RB(c,c),
                        RB(c,c),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x= F.interpolate(x,scale_factor=2.0, mode='bilinear',align_corners=False)
        x= self.up0_(x)
    
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x= F.interpolate(x,scale_factor=2.0, mode='bilinear',align_corners=False)
        x=self.up1_(x)
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)
            

class dec_dq_opt(nn.Module):
    def __init__(self, c, out=3):
        super(dec_dq_opt, self).__init__()
        print("dec_dq_opt")
    
        self.up0 = nn.Sequential(
                         mlp(17+4*c,4*c*4, 4*c),
                         Conv2(4*c,4*c,3,1,1),
                         Conv2(4*c,4*c,3,1,1),
                         deconv(4*c, 2*c)
        )
        
        self.up1 =  nn.Sequential(
                        mlp(17+2*c+2*c,2*c*4, 2*c),
                         Conv2(2*c,2*c,3,1,1),
                         Conv2(2*c,2*c,3,1,1),
                        deconv(2*c, c)
        )
        self.up2 = nn.Sequential(
                        mlp(17+c+c,c*4,c),
                         Conv2(c,c,3,1,1),
                         Conv2(c,c,3,1,1),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)
            



class r(nn.Module):
    def __init__(self, c, out=3):
        super(r, self).__init__()
        print("r")
    
        self.up0 = nn.Sequential(
                         mlp(17+4*c,4*c*4, 4*c),
                         deconv(4*c, 2*c)
        )
        
        self.up1 =  nn.Sequential(
                        mlp(17+2*c+2*c,2*c*4, 2*c),
                        deconv(2*c, c)
        )
        self.up2 = nn.Sequential(
                        mlp(17+c+c,c*4,c),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)
            

class r2(nn.Module):
    def __init__(self, c, out=3):
        super(r2, self).__init__()
        print("r2")
    
        self.up0 = nn.Sequential(
                         Conv2(17+4*c, 4*c,3,1,1),
                         deconv(4*c, 2*c)
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(17+2*c+2*c, 2*c,3,1,1),
                        deconv(2*c, c)
        )
        self.up2 = nn.Sequential(
                        Conv2(17+c+c,c,3,1,1),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)
            

class r22(nn.Module):
    def __init__(self, c, out=3):
        super(r22, self).__init__()
        print("r22")
    
        self.up0 = nn.Sequential(
                         Conv2(17+4*c, 4*c,3,1,1),
                         Conv2(4*c, 4*c,3,1,1),
                         deconv(4*c, 2*c)
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(17+2*c+2*c, 2*c,3,1,1),
                        Conv2(2*c, 2*c,3,1,1),
                        deconv(2*c, c)
        )
        self.up2 = nn.Sequential(
                        Conv2(17+c+c,c,3,1,1),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)
            
class r3(nn.Module):
    def __init__(self, c, out=3):
        super(r3, self).__init__()
    

        self.up0 = nn.Sequential(
                         Conv2(17+4*c,4*c,3,1,1),
                         Conv2(4*c,4*c,3,1,1),
                         deconv(4*c,2*c),
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(4*c, 2*c,3,1,1),
                        Conv2(2*c, 2*c,3,1,1),
                        deconv(2*c,c),
        )
        self.up2 = nn.Sequential(
                        Conv2(c+c, c,3,1,1),
                        Conv2(c, c,3,1,1),
                        nn.Conv2d(c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat(( imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, Cs[-2]), 1))
        x = self.up2(torch.cat((x, Cs[-3]), 1))
        return torch.sigmoid(x)
    
class r4(nn.Module):
    def __init__(self, c, out=3):
        super(r4, self).__init__()
        print('r4')    
        self.up0 = nn.Sequential(
                         Conv2(17+4*c,4*c,3,1,1),
                         deconv(4*c,2*c),
        )

        self.conv1=  conv(2*c, 2*c,3,1,1)
        self.up1 =  nn.Sequential(
                        deconv(4*c,c),
        )

        self.conv2=  conv(c, c,3,1,1)
        self.up2 = nn.Sequential(
                        nn.Conv2d(2*c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat([x,self.conv1( Cs[-2])], 1))
        x = self.up2(torch.cat((x,self.conv2( Cs[-3])), 1))
        return torch.sigmoid(x)
    
class r5(nn.Module):
    def __init__(self, c, out=3):
        super(r5, self).__init__()
        print('r5')    
        self.up0 = nn.Sequential(
                         Conv2(17+4*c,4*c,3,1,1),
                         deconv(4*c,2*c),
        )

        self.conv1=  Conv2(2*c, 2*c,3,1,1)
        self.up1 =  nn.Sequential(
                        deconv(4*c,c),
        )

        self.conv2=  Conv2(c, c,3,1,1)
        self.up2 = nn.Sequential(
                        nn.Conv2d(2*c , out, 3, 1, 1)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat([x,self.conv1( Cs[-2])], 1))
        x = self.up2(torch.cat((x,self.conv2( Cs[-3])), 1))
        return torch.sigmoid(x)
    

class r_cd(nn.Module):
    def __init__(self, c, out=3):
        super(r_cd, self).__init__()
        print("r_cd")
    
        self.up0 = nn.Sequential(
                         conv(17+4*c, 4*c),
                         deconv(4*c, 2*c)
        )
        
        self.up1 =  nn.Sequential(
                        conv(17+2*c+2*c, 2*c),
                        deconv(2*c, c)
        )
        self.up2 = nn.Sequential(
                        conv(17+c+c,c),
                        nn.Conv2d(c , out, 1, 1, 0)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)
            




class r_c2d(nn.Module):
    def __init__(self, c, out=3):
        super(r_c2d, self).__init__()
        print("r_c2d")
    
        self.up0 = nn.Sequential(
                         Conv2(17+4*c, 4*c),
                         deconv(4*c, 2*c)
        )
        
        self.up1 =  nn.Sequential(
                        Conv2(17+2*c+2*c, 2*c),
                        deconv(2*c, c)
        )
        self.up2 = nn.Sequential(
                        Conv2(17+c+c,c),
                        nn.Conv2d(c , out, 1, 1, 0)
        )
        
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

    def forward(self, imgs, wimgs, masks, flows, Cs):
        x = self.up0(torch.cat((   imgs[-1],wimgs[-1],Cs[-1],flows[-1],masks[-1]), 1)) # +17 + 8c
        x = self.up1(torch.cat((x, imgs[-2],wimgs[-2],Cs[-2],flows[-2],masks[-2]), 1))
        x = self.up2(torch.cat((x, imgs[-3],wimgs[-3],Cs[-3],flows[-3],masks[-3]), 1))
        return torch.sigmoid(x)
            


