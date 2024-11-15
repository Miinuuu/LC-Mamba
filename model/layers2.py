import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from .warplayer import * 
from .convnext import * 
from .cbam import * 
from .layers import * 
#from .mobileone import *
#from .replknet import *
#from .fastvit import *
from .transforms import *
##############################################################################################################################################################################################################################
##############################################################################################################################################################################################################################


def convbn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
                  torch.nn.BatchNorm2d(out_planes),
                  nn.PReLU(out_planes)
        )
    

def deconvbn(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        torch.nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
        )

def Dwconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,groups=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True,groups=groups),
        nn.PReLU(out_planes)
        )


def pwconv(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )

def Dwconvbn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,groups=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True,groups=groups),
        torch.nn.BatchNorm2d(out_planes),

        nn.PReLU(out_planes))

class Conv2bn(nn.Module):
    def __init__(self, in_planes, out_planes,kernel_size=3, stride=2,padding=1):
        super(Conv2bn, self).__init__()
        self.conv1 = nn.Sequential(convbn)
        self.conv2 = nn.Sequential(convbn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
        

    
class ConvRBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2,act_layer=nn.PReLU):
        super().__init__()

        self.conv2 = nn.Sequential(
            conv(in_dim,out_dim),
            nn.Conv2d(out_dim,out_dim,3,1,1)
        )
        self.act = nn.PReLU(out_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        return self.act(x+self.conv2(x))


class ConvBlockbn(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2,act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(convbn(in_dim, out_dim, 3,1,1))
            else:
                layers.append(convbn(out_dim, out_dim, 3,1,1))
       
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



class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int,depths=2,act_layer=nn.PReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch,3,1,1)
        self.PReLU1 = nn.PReLU(out_ch)
        self.PReLU2 = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch,3,1,1)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch,1,1,0)
        else:
            self.skip = None

    def forward(self, x) :
        identity = x

        out = self.conv1(x)
        out = self.PReLU1(out)
        out = self.conv2(out)
        out = self.PReLU2(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out
    
class DWConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim,kernel_size=7,padding=3, depths=2,act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(Dwconv(in_dim, in_dim,kernel_size,1,padding='same',groups=in_dim))
                layers.append(pwconv(in_dim, out_dim, 1))
            else:
                layers.append(Dwconv(out_dim, out_dim, kernel_size ,1,padding='same',groups=out_dim))
                layers.append(pwconv(out_dim, out_dim, 1))

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


class OverlapPatchEmbedbn(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2)),nn.BatchNorm2d(embed_dim))
        self.norm = nn.BatchNorm2d(embed_dim)

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



class OverlapPatchEmbed_opt(nn.Module):
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


class OverlapPatchEmbed_dw(nn.Module):
    def __init__(self, patch_size=7, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, in_chans, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2),groups=in_chans)
        self.pwconv = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1,
                              padding=0)
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
        x=self.pwconv(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.prelu(x + self.conv5(out))
        return out

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,groups=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True,groups=groups),
        nn.PReLU(out_planes)
        )



##############################################################################################################################################################################################################################
##############################################################################################################################################################################################################################


class ConvGRU(nn.Module):
    def __init__(self, input_dim=192+128,hidden_dim=128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)

    def forward(self, h,x):
        hx = torch.cat((h,x),1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h
    
class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow
    


class IFBlock(nn.Module):
    def __init__(self, in_planes, c, scale):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        self.scale = scale

    def forward(self, x, flow):
        scale = self.scale
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
        x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask
    






class CrossScalePatchEmbed_dw(nn.Module):
    def __init__(self, in_dims=[16,32,64], embed_dim=768,k=3):
        super().__init__()
        print('CrossScalePatchEmbed_dw')
        base_dim = in_dims[0]

        self.k=k
        layers = []
        for i in range(len(in_dims)):
            for j in range(2 ** i):
                k= self.k + (j)* (self.k-1)  
                #layers.append(nn.Conv2d(in_dims[-1-i], base_dim, 3, 2**(i+1), 1+j, 1+j))
                layers.append(nn.Conv2d(in_dims[-1-i], 
                                        base_dim, 
                                        kernel_size=k ,
                                        stride=2**(i+1),
                                        padding= k//2, 
                                        #groups = in_dims[-1-i]))
                                        groups = base_dim))
                
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Conv2d(base_dim * len(layers), embed_dim , 1, 1) # 112 -> 128
        self.norm = nn.LayerNorm(embed_dim)

        #self.pwconv1 = nn.Linear(embed_dim, 4 * embed_dim) # pointwise/1x1 convs, implemented with linear layers
        #self.pwconv2 = nn.Linear( 4*embed_dim,embed_dim)

        #self.act = nn.GELU()
        #self.gamma = nn.Parameter(1e-6 * torch.ones((embed_dim)), requires_grad=True)
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
        
        x = self.proj(torch.cat(ys,1)) # embed dim
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # B,N,C
        x = self.norm(x)
   
        return x, H, W


class CrossScalePatchEmbed_dwbt(nn.Module):
    def __init__(self, in_dims=[16,32,64], embed_dim=768,k=3):
        super().__init__()
        print('CrossScalePatchEmbed_dwbt')
        base_dim = in_dims[0]

        self.k=k
        layers = []
        for i in range(len(in_dims)):
            for j in range(2 ** i):
                #layers.append(nn.Conv2d(in_dims[-1-i], base_dim, 3, 2**(i+1), 1+j, 1+j))
                k=self.k + (j)* (self.k-1)
                layers.append(nn.Conv2d(in_dims[-1-i], 
                                        base_dim, 
                                        kernel_size=k,
                                        stride=2**(i+1),
                                        padding= k//2, 
                                        #groups = in_dims[-1-i]))
                                        groups = base_dim))
                
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Conv2d(base_dim * len(layers), embed_dim , 1, 1) # 112 -> 128
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
            x=[]
            for _ in range(2 ** i):
                x.append(self.layers[k](xs[-1-i]))
                k += 1
            ys.append(torch.cat(x,1))
        
        x = self.proj(torch.cat(ys,1)) # embed dim
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # B,N,C
        x = self.norm(x)
   
        ys[0]= F.interpolate(ys[0],scale_factor=4)
        ys[1]= F.interpolate(ys[1],scale_factor=2)
        
        return x, ys,H, W


class CrossScalePatchEmbed_dwbt2(nn.Module):
    def __init__(self, in_dims=[16,32,64], embed_dim=768):
        super().__init__()
        print('CrossScalePatchEmbed_dwbt2')
        base_dim = in_dims[0]

        self.k=3
        layers0 = []
        layers = []



        for i in range(len(in_dims)):
            #k= 2**(i+1) +1 # 3,5,9
            k= 3
            layers0.append( nn.Sequential(nn.Conv2d(in_dims[-1-i], 
                                                    in_dims[-1-i], 
                                                    kernel_size= k,
                                                    stride=1,
                                                    padding= k//2, ), 
                                                    nn.PReLU(in_dims[-1-i])))

            for j in range(2 ** i):
                #layers.append(nn.Conv2d(in_dims[-1-i], base_dim, 3, 2**(i+1), 1+j, 1+j))
                layers.append(nn.Conv2d(in_dims[-1-i], 
                                        base_dim, 
                                        kernel_size= self.k + (j)* (self.k-1) ,
                                        stride=2**(i+1),
                                        padding= 1+j, 
                                        #groups = in_dims[-1-i]))
                                        groups = base_dim))
                
        self.layers = nn.ModuleList(layers)
        self.layers0 = nn.ModuleList(layers0)
        self.proj = nn.Conv2d(base_dim * len(layers), embed_dim , 1, 1) # 112 -> 128
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
        ys0 = []
        ys = []
        k = 0
        for i in range(len(xs)):
            ys0.append(self.layers0[i](xs[-1-i]))
            x=[]
            for _ in range(2 ** i):
                x.append(self.layers[k](ys0[-1]))
                #print(k,x[-1].shape)
                k += 1
            ys.append(torch.cat(x,1))
        
        x = self.proj(torch.cat(ys,1)) # embed dim
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # B,N,C
        x = self.norm(x)
   
        return x,ys0, H, W

class CrossScalePatchEmbed_bt3(nn.Module):
    def __init__(self, in_dims=[16,32,64], embed_dim=768):
        super().__init__()
        print('CrossScalePatchEmbed_bt3')
        base_dim = in_dims[0]

        self.k=3
        layers0 = []
        layers = []



        for i in range(len(in_dims)):
            #k= 2**(i+1) +1 # 3,5,9
            k= 3
            layers0.append( nn.Sequential(nn.Conv2d(in_dims[-1-i], 
                                                    in_dims[-1-i], 
                                                    kernel_size= k,
                                                    stride=1,
                                                    padding= k//2, ), 
                                                    nn.PReLU(in_dims[-1-i])))

                #layers.append(nn.Conv2d(in_dims[-1-i], base_dim, 3, 2**(i+1), 1+j, 1+j))
            k= 2**(i+1) +1 # 3,5,9
            layers.append(nn.Conv2d(    in_dims[-1-i], 
                                        in_dims[-1-i], 
                                        kernel_size= k ,
                                        stride=2**(i+1),
                                        padding= k//2, 
                                        #groups = in_dims[-1-i]))
                                        ))
                
        self.layers = nn.ModuleList(layers)
        self.layers0 = nn.ModuleList(layers0)
        self.proj = nn.Conv2d(in_dims[0]+in_dims[1]+in_dims[2], embed_dim , 1, 1) # 112 -> 128
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
        ys0 = []

        x=[]
        for i in range(len(xs)):
            ys0.append(self.layers0[i](xs[-1-i]))
            x.append(self.layers[i](ys0[-1]))
        
        x = self.proj(torch.cat(x,1)) # embed dim
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # B,N,C
        x = self.norm(x)
   
        return x,ys0, H, W
class CrossScalePatchEmbed_dw_opt(nn.Module):
    def __init__(self, in_dims=[16,32,64], embed_dim=768):
        super().__init__()
        print('CrossScalePatchEmbed_dw_opt')
        base_dim = in_dims[0]

        self.k=3
        layers = []
        for i in range(len(in_dims)):
            for j in range(2 ** i):
                #layers.append(nn.Conv2d(in_dims[-1-i], base_dim, 3, 2**(i+1), 1+j, 1+j))
                layers.append(nn.Conv2d(in_dims[-1-i], 
                                        base_dim, 
                                        kernel_size= self.k + (j)* (self.k-1) ,
                                        stride=2**(i+1),
                                        padding= 1+j, 
                                        #groups = in_dims[-1-i]))
                                        groups = base_dim))
                
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Conv2d(base_dim * len(layers), embed_dim , 1, 1) # 112 -> 128
        self.norm = nn.LayerNorm(embed_dim)

        #self.pwconv1 = nn.Linear(embed_dim, 4 * embed_dim) # pointwise/1x1 convs, implemented with linear layers
        #self.pwconv2 = nn.Linear( 4*embed_dim,embed_dim)

        #self.act = nn.GELU()
        #self.gamma = nn.Parameter(1e-6 * torch.ones((embed_dim)), requires_grad=True)
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
        
        x = self.proj(torch.cat(ys,1)) # embed dim
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # B,N,C
        x = self.norm(x)
   
        return x, H, W



class CrossScalePatchEmbed_dwbn(nn.Module):
    def __init__(self, in_dims=[16,32,64], embed_dim=768):
        super().__init__()
        print('CrossScalePatchEmbed_dwbn')
        base_dim = in_dims[0]

        self.k=3
        layers = []
        for i in range(len(in_dims)):
            for j in range(2 ** i):
                #layers.append(nn.Conv2d(in_dims[-1-i], base_dim, 3, 2**(i+1), 1+j, 1+j))
                layers.append( nn.Sequential(
                                        nn.Conv2d(in_dims[-1-i], 
                                        base_dim, 
                                        kernel_size= self.k + (j)* (self.k-1) ,
                                        stride=2**(i+1),
                                        padding= 1+j, 
                                        #groups = in_dims[-1-i]))
                                        groups = base_dim),
                                        nn.BatchNorm2d(base_dim),))
                
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Sequential(nn.Conv2d(base_dim * len(layers), embed_dim , 1, 1),nn.BatchNorm2d(embed_dim))# 112 -> 128
        self.norm = nn.BatchNorm2d(embed_dim)

        #self.pwconv1 = nn.Linear(embed_dim, 4 * embed_dim) # pointwise/1x1 convs, implemented with linear layers
        #self.pwconv2 = nn.Linear( 4*embed_dim,embed_dim)

        #self.act = nn.GELU()
        #self.gamma = nn.Parameter(1e-6 * torch.ones((embed_dim)), requires_grad=True)
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
        
        x = self.proj(torch.cat(ys,1)) # embed dim
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # B,N,C
        x = self.norm(x)
   
        return x, H, W

def mlp(in_c,h_c,out_c,k=1,s=1,p=0):
    return nn.Sequential(
        nn.Conv2d(in_c,h_c,1,1,0),
        nn.PReLU(h_c),
        nn.Conv2d(h_c,out_c,1,1,0)
        )

class RB(nn.Module):
    def __init__(self,c_i,c_o,k=3,s=1,p=1) -> None:
        super().__init__()
        self.conv2 = nn.Sequential(
            conv(c_i,c_o),
            nn.Conv2d(c_o,c_o,k,s,p)
        )
        self.act = nn.PReLU(c_o)
   
    def forward(self, x):
        return self.act(x+self.conv2(x))
    
class ConvBlock_RB(nn.Module):
    def __init__(self,in_ch,out_ch,depth) -> None:
        super().__init__()
        self.conv2 = nn.Sequential(
            conv(in_ch,out_ch),
            nn.Conv2d(out_ch,out_ch,3,1,1)
        )
        self.act = nn.PReLU(out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch,1,1,0)
        else:
            self.skip = None


    def forward(self, x):
        shortcut=x
        x=self.conv2(x)
        
        if self.skip is not None:
            shortcut = self.skip(shortcut)
        x= self.act(x+shortcut)
        return x
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)
    

def upsample_flow( flow, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = flow.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(8 * flow, [3, 3], padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, 2, 8 * H, 8 * W)


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y
    
class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, flow, mask):
        flow = self.se_rgb(flow)
        mask = self.se_depth(mask)
        out = flow + mask
        return out

def convex_upsample(x, mask, upscale=8, kernel=3):
    """ Upsample x [B, C, H/d, W/d] -> [B, C, H, W] using convex combination """
    B, C, H, W = x.shape
    mask = mask.view(B, 1, kernel*kernel, upscale, upscale, H, W)

    mask = torch.softmax(mask, dim=2)
    x = F.unfold(x, [kernel,kernel], padding=kernel//2)
    x = x.view(B, C, kernel*kernel, 1, 1, H, W)
    x = torch.sum(mask * x, dim=2) # B,C,upscale,upscale,H,W
    x = x.permute(0, 1, 4, 2, 5, 3)  # B,C,H,upscale,W,upscale

    return x.reshape(B, C, H*upscale, W*upscale)


#####################################################################################################
class Head_bi(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head_bi, self).__init__()
        #self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
                                  conv(in_planes*2  + in_else, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  

    def forward(self, motion_feature, x, flow): # /16 /8 /4
        #motion_feature = self.upsample(motion_feature) #/4 /2 /1
        motion_feature = F.interpolate(motion_feature,scale_factor=4.0 , mode= "bilinear", align_corners=False) #/4 /2 /1

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
    

class Head_opt(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head_opt, self).__init__()
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
    


class Head_pme(nn.Module):
    def __init__(self, in_planes, scale,c ,last):
        super(Head_pme, self).__init__()

        self.scale = scale
        self.last= last
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PixelShuffle(2))

        self.bodyconv = nn.Sequential(
            conv(in_planes*2//16 +12+5+1, c, 3, 1,1),
            conv(c, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
                                  ) 
        self.flowconv = nn.Sequential(
            conv(c, 5, 3, 1, 1))
        
        if self.last:
            self.resconv = nn.Sequential(Conv2(c+c//2+c,c,3,1,1),Conv2(c,c,3,1,1),deconv(c,c//2))  # 2c *2 + res_i-1 + c
            self.lastconv= nn.Sequential(Conv2(c//2+c//2,c,3,1,1),nn.Conv2d(c,3,3,1,1)) #4c //2 =2c
        else:
            self.resconv = nn.Sequential(Conv2(2*c+c,c,3,1,1),Conv2(c,c,3,1,1),deconv(c,c//2)) 

    def forward(self,a0,a1,stem,res,flow,mask,img0,img1,timestep): # /16 /8 /4
        B,C,H,W=a0.shape
        if self.scale > 1.0:
            a0= self.upsample(a0)
            a1= self.upsample(a1)

        if img0.shape[-1] > a0.shape[-1]:
            img0=F.interpolate(img0,size= a0.shape[-2:],mode="bilinear", align_corners=False)
            img1=F.interpolate(img1,size= a1.shape[-2:],mode="bilinear", align_corners=False)
        
        t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
        
        if flow != None:
            flow = F.interpolate(flow,scale_factor=2.0,mode="bilinear",align_corners=False)*(2.0)
            mask = F.interpolate(mask,scale_factor=2.0,mode="bilinear",align_corners=False)
            wimg0    =  backward_warp(img0,flow[:,0:2])
            wimg1    =  backward_warp(img1,flow[:,2:4])
        
        if flow == None:
            wimg0=img0
            wimg1=img1
            flow = torch.ones_like(a0[:,:4],device=a0.device)*1e-6
            mask = torch.ones_like(a0[:,:1],device=a0.device)*0.5

        x = torch.cat((a0,a1,img0,img1,wimg0,wimg1,flow,mask,t),dim=1)
        x_ = self.bodyconv(x)
        x =  self.flowconv(x_)
        flow = x[:,:4]  + flow
        mask = x[:,4:5] + mask 


        if res == None:
            res = self.resconv(torch.cat( (backward_warp(stem[2][:B],flow[:,:2]),
                                           backward_warp(stem[2][B:],flow[:,2:4]),
                                           x_),1))
        else:
            res = self.resconv(torch.cat((
                                          backward_warp(stem[1][:B],flow[:,:2]),
                                          backward_warp(stem[1][B:],flow[:,2:4]),
                                          x_,res),1))
    
        if self.scale >1.0:
            if self.scale != 4:
                flow_up=F.interpolate(flow,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)*(self.scale // 4)
                mask_up=F.interpolate(mask,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)

        if self.last:
            res = self.lastconv(torch.cat((backward_warp(stem[0][:B],flow_up[:,:2]),
                                           backward_warp(stem[0][B:],flow_up[:,2:4])
                                           ,res),1))


        return flow_up,mask_up,flow, mask,res
    
    


class Head_uniopt(nn.Module):
    def __init__(self, in_planes, scale,c ,last):
        super(Head_uniopt, self).__init__()

        self.scale = scale
        self.last= last
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PixelShuffle(2))

        #self.first_conv = conv(in_planes*2,c,3,1,1)
        self.ca = ChannelAttention(in_planes*2,8)
        self.bodyconv = nn.Sequential(
            conv(in_planes*2//16 +12+5+1,2, 3, 1,1),
            conv(2, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
                                  ) 
        self.flowconv = nn.Sequential(
            conv(c, 5, 3, 1, 1))
        
        if self.last:
            self.resconv = nn.Sequential(conv(c+c//2+c,2*c,3,1,1), ChannelAttention(2*c,8),nn.PixelShuffle(2), conv(c//2,c//2))  # 2c *2 + res_i-1 + c
            self.lastconv= nn.Sequential(conv(c//2+c//2,c,3,1,1),nn.Conv2d(c,3,3,1,1)) #4c //2 =2c
        else:
            self.resconv = nn.Sequential(conv(4*c*2 //4+ c ,2*c,3,1,1),ChannelAttention(2*c,8),nn.PixelShuffle(2),conv(c//2,c//2)) 
    
    def forward(self,a0,a1,stem,res,flow,mask,img0,img1,timestep): # /16 /8 /4
        B,C,H,W=a0.shape
        if self.scale > 1.0:
            a= self.upsample(self.ca(torch.cat((a0,a1),1)))


        if img0.shape[-1] > a0.shape[-1]:
            img0=F.interpolate(img0,size= a.shape[-2:],mode="bilinear", align_corners=False)
            img1=F.interpolate(img1,size= a.shape[-2:],mode="bilinear", align_corners=False)
        
        t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
        
        if flow != None:
            flow = F.interpolate(flow,scale_factor=2.0,mode="bilinear",align_corners=False)*(2.0)
            mask = F.interpolate(mask,scale_factor=2.0,mode="bilinear",align_corners=False)
            wimg0    =  backward_warp(img0,flow[:,0:2])
            wimg1    =  backward_warp(img1,flow[:,2:4])
        
        if flow == None:
            wimg0=img0
            wimg1=img1
            flow = torch.ones_like(a[:,:4],device=a.device)*1e-6
            mask = torch.ones_like(a[:,:1],device=a.device)*0.5

        x = torch.cat((a,img0,img1,wimg0,wimg1,flow,mask,t),dim=1)
        x_ = self.bodyconv(x)
        x =  self.flowconv(x_)
        flow = x[:,:4]  + flow
        mask = x[:,4:5] + mask 


        if res == None:
            res = self.resconv(torch.cat( (backward_warp(stem[2][:B],flow[:,:2]),
                                           backward_warp(stem[2][B:],flow[:,2:4]),
                                           x_),1))
        else:
            res = self.resconv(torch.cat((
                                          backward_warp(stem[1][:B],flow[:,:2]),
                                          backward_warp(stem[1][B:],flow[:,2:4]),
                                          x_,res),1))
    
        if self.scale >1.0:
            if self.scale != 4:
                flow_up=F.interpolate(flow,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)*(self.scale // 4)
                mask_up=F.interpolate(mask,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)

        if self.last:
            res = self.lastconv(torch.cat((backward_warp(stem[0][:B],flow_up[:,:2]),
                                           backward_warp(stem[0][B:],flow_up[:,2:4])
                                           ,res),1))


        return flow_up,mask_up,flow, mask,res


class Head_uniopt_bi(nn.Module):
    def __init__(self, in_planes, scale,c ,last):
        super(Head_uniopt_bi, self).__init__()

        self.scale = scale
        self.last= last
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PixelShuffle(2))

        #self.first_conv = conv(in_planes*2,c,3,1,1)
        self.ca = ChannelAttention(in_planes*2,8)
        
        if self.last:
            self.bodyconv = nn.Sequential(
                                      conv(in_planes*2//16 +12+5+1 + 3 ,c, 3, 1,1),
                                      conv(c, c, 3, 1, 1),
                                      conv(c, c, 3, 1, 1),
                                      conv(c, 5+3, 3, 1, 1))
        else:

            self.bodyconv = nn.Sequential(
                                      conv(in_planes*2//16 +12+5+1 + 3, c, 3, 1,1),
                                      conv(c, c, 3, 1, 1),
                                      conv(c, c, 3, 1, 1),
                                      conv(c, 5+3, 3, 1, 1),
                                      )
        

    def forward(self,a0,a1,stem,res,res_up,flow,mask,img0,img1,timestep): # /16 /8 /4
        B,C,H,W=a0.shape
        if self.scale > 1.0:
            a= self.upsample(self.ca(torch.cat((a0,a1),1)))


        if img0.shape[-1] > a0.shape[-1]:
            img0=F.interpolate(img0,size= a.shape[-2:],mode="bilinear", align_corners=False)
            img1=F.interpolate(img1,size= a.shape[-2:],mode="bilinear", align_corners=False)
        
        t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
        
        if flow != None:
            flow = F.interpolate(flow,scale_factor=2.0,mode="bilinear",align_corners=False)*(2.0)
            mask = F.interpolate(mask,scale_factor=2.0,mode="bilinear",align_corners=False)
            res = F.interpolate(res,scale_factor=2.0,mode="bilinear",align_corners=False)
            wimg0    =  backward_warp(img0,flow[:,0:2])
            wimg1    =  backward_warp(img1,flow[:,2:4])
        
        if flow == None:
            wimg0=img0
            wimg1=img1
            flow = torch.ones_like(a[:,:4],device=a.device)*1e-6
            res = torch.ones_like(a[:,:3],device=a.device)*1e-6
            mask = torch.ones_like(a[:,:1],device=a.device)*0.5

        x = torch.cat((a,img0,img1,wimg0,wimg1,flow,mask,res,t),dim=1)
        x = self.bodyconv(x)
        flow = x[:,:4]  + flow
        mask = x[:,4:5] + mask 
        res= x[:,5:] + res
        
    
        if self.scale >1.0:
            if self.scale != 4:
                flow_up=F.interpolate(flow,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)*(self.scale // 4)
                mask_up=F.interpolate(mask,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)
                res_up=F.interpolate(res,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)

  
        return flow_up,mask_up, res_up, flow, mask,res
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################      
    
class Head_pme_ca(nn.Module):
    def __init__(self, in_planes, scale,c ,last):
        super(Head_pme_ca, self).__init__()

        self.scale = scale
        self.last= last
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PixelShuffle(2))

        #self.first_conv = conv(in_planes*2,c,3,1,1)
        self.ca = ChannelAttention(in_planes*2,16)
        self.bodyconv = nn.Sequential(
            conv(in_planes*2//16 +12+5+1,c, 3, 1,1),
            conv(c, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
                                  ) 
        self.flowconv = nn.Sequential(
            conv(c, 5, 3, 1, 1))
        
        if self.last:
            self.resconv = nn.Sequential(mlp(c+c//2+c,c,3,1,1), RB(c,c),RB(c,c),deconv(c,c//2))  # 2c *2 + res_i-1 + c
            self.lastconv= nn.Sequential(mlp(c//2+c//2,c,3,1,1),RB(c,c),RB(c,c),nn.Conv2d(c,3,3,1,1)) #4c //2 =2c
        else:
            self.resconv = nn.Sequential(mlp(2*c+c,c,3,1,1),RB(c,c),RB(c,c),deconv(c,c//2)) 
    
    def forward(self,a0,a1,stem,res,flow,mask,img0,img1,timestep): # /16 /8 /4
        B,C,H,W=a0.shape
        if self.scale > 1.0:
            a= self.upsample(self.ca(torch.cat((a0,a1),1)))


        if img0.shape[-1] > a0.shape[-1]:
            img0=F.interpolate(img0,size= a.shape[-2:],mode="bilinear", align_corners=False)
            img1=F.interpolate(img1,size= a.shape[-2:],mode="bilinear", align_corners=False)
        
        t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
        
        if flow != None:
            flow = F.interpolate(flow,scale_factor=2.0,mode="bilinear",align_corners=False)*(2.0)
            mask = F.interpolate(mask,scale_factor=2.0,mode="bilinear",align_corners=False)
            wimg0    =  backward_warp(img0,flow[:,0:2])
            wimg1    =  backward_warp(img1,flow[:,2:4])
        
        if flow == None:
            wimg0=img0
            wimg1=img1
            flow = torch.ones_like(a[:,:4],device=a.device)*1e-6
            mask = torch.ones_like(a[:,:1],device=a.device)*0.5

        x = torch.cat((a,img0,img1,wimg0,wimg1,flow,mask,t),dim=1)
        x_ = self.bodyconv(x)
        x =  self.flowconv(x_)
        flow = x[:,:4]  + flow
        mask = x[:,4:5] + mask 


        if res == None:
            res = self.resconv(torch.cat( (backward_warp(stem[2][:B],flow[:,:2]),
                                           backward_warp(stem[2][B:],flow[:,2:4]),
                                           x_),1))
        else:
            res = self.resconv(torch.cat((
                                          backward_warp(stem[1][:B],flow[:,:2]),
                                          backward_warp(stem[1][B:],flow[:,2:4]),
                                          x_,res),1))
    
        if self.scale >1.0:
            if self.scale != 4:
                flow_up=F.interpolate(flow,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)*(self.scale // 4)
                mask_up=F.interpolate(mask,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)

        if self.last:
            res = self.lastconv(torch.cat((backward_warp(stem[0][:B],flow_up[:,:2]),
                                           backward_warp(stem[0][B:],flow_up[:,2:4])
                                           ,res),1))


        return flow_up,mask_up,flow, mask,res
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################      
class Head_pme_ca2(nn.Module):
    def __init__(self, in_planes, scale,c ,last):
        super(Head_pme_ca2, self).__init__()

        self.scale = scale
        self.last= last
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PixelShuffle(2))

        #self.first_conv = conv(in_planes*2,c,3,1,1)
        self.ca = ChannelAttention(in_planes*2,16)
        self.bodyconv = nn.Sequential(
            conv(in_planes*2//16 +12+5+1,c, 3, 1,1),
            conv(c, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
                                  ) 
        self.flowconv = nn.Sequential(
            conv(c, 5, 3, 1, 1))
        
        if self.last:
            self.resconv = nn.Sequential(Conv2(c+c//2+c + 5,c,3,1,1),deconv(c,c//2))  # 2c *2 + res_i-1 + c
            self.lastconv= nn.Sequential(Conv2(c//2+c//2 + 5 ,c,3,1,1),nn.Conv2d(c,3,3,1,1)) #4c //2 =2c
        else:
            self.resconv = nn.Sequential(Conv2(2*c+c + 5 ,c,3,1,1),deconv(c,c//2)) 
    
    def forward(self,a0,a1,stem,res,flow,mask,img0,img1,timestep): # /16 /8 /4
        B,C,H,W=a0.shape
        if self.scale > 1.0:
            a= self.upsample(self.ca(torch.cat((a0,a1),1)))


        if img0.shape[-1] > a0.shape[-1]:
            img0=F.interpolate(img0,size= a.shape[-2:],mode="bilinear", align_corners=False)
            img1=F.interpolate(img1,size= a.shape[-2:],mode="bilinear", align_corners=False)
        
        t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
        
        if flow != None:
            flow = F.interpolate(flow,scale_factor=2.0,mode="bilinear",align_corners=False)*(2.0)
            mask = F.interpolate(mask,scale_factor=2.0,mode="bilinear",align_corners=False)
            wimg0    =  backward_warp(img0,flow[:,0:2])
            wimg1    =  backward_warp(img1,flow[:,2:4])
        
        if flow == None:
            wimg0=img0
            wimg1=img1
            flow = torch.ones_like(a[:,:4],device=a.device)*1e-6
            mask = torch.ones_like(a[:,:1],device=a.device)*0.5

        x = torch.cat((a,img0,img1,wimg0,wimg1,flow,mask,t),dim=1)
        x_ = self.bodyconv(x)
        x =  self.flowconv(x_)
        flow = x[:,:4]  + flow
        mask = x[:,4:5] + mask 


        if res == None:
            res = self.resconv(torch.cat( (backward_warp(stem[2][:B],flow[:,:2]),
                                           backward_warp(stem[2][B:],flow[:,2:4]),
                                           flow,
                                           mask,
                                           x_),1))
        else:
            res = self.resconv(torch.cat((
                                          backward_warp(stem[1][:B],flow[:,:2]),
                                          backward_warp(stem[1][B:],flow[:,2:4]),
                                          flow,
                                          mask,
                                          x_,res),1))
    
        if self.scale >1.0:
            if self.scale != 4:
                flow_up=F.interpolate(flow,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)*(self.scale // 4)
                mask_up=F.interpolate(mask,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)

        if self.last:
            res = self.lastconv(torch.cat((backward_warp(stem[0][:B],flow_up[:,:2]),
                                           backward_warp(stem[0][B:],flow_up[:,2:4]),
                                           flow_up,
                                           mask_up,
                                           res),1))


        return flow_up,mask_up,flow, mask,res
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################      
class Head_pme_ca2_1(nn.Module):
    def __init__(self, in_planes, scale,c ,last):
        super(Head_pme_ca2_1, self).__init__()

        self.scale = scale
        self.last= last
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PixelShuffle(2))

        #self.first_conv = conv(in_planes*2,c,3,1,1)
        self.ca = ChannelAttention(in_planes*2,16)
        self.bodyconv = nn.Sequential(
            conv(in_planes*2//16 +12+5+1,c, 3, 1,1),
            conv(c, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
                                  ) 
        self.flowconv = nn.Sequential(
            conv(c, 5, 3, 1, 1))
        
        if self.last:
            self.resconv = nn.Sequential(Conv2(c+c//2+c + 5,c,3,1,1),deconv(c,c//2))  # 2c *2 + res_i-1 + c
            self.lastconv= nn.Sequential(Conv2(c//2 + 5 ,c,3,1,1),nn.Conv2d(c,3,3,1,1)) #4c //2 =2c
        else:
            self.resconv = nn.Sequential(Conv2(2*c+c + 5 ,c,3,1,1),deconv(c,c//2)) 
    
    def forward(self,a0,a1,stem,res,flow,mask,img0,img1,timestep): # /16 /8 /4
        B,C,H,W=a0.shape
        if self.scale > 1.0:
            a= self.upsample(self.ca(torch.cat((a0,a1),1)))


        if img0.shape[-1] > a0.shape[-1]:
            img0=F.interpolate(img0,size= a.shape[-2:],mode="bilinear", align_corners=False)
            img1=F.interpolate(img1,size= a.shape[-2:],mode="bilinear", align_corners=False)
        
        t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
        
        if flow != None:
            flow = F.interpolate(flow,scale_factor=2.0,mode="bilinear",align_corners=False)*(2.0)
            mask = F.interpolate(mask,scale_factor=2.0,mode="bilinear",align_corners=False)
            wimg0    =  backward_warp(img0,flow[:,0:2])
            wimg1    =  backward_warp(img1,flow[:,2:4])
        
        if flow == None:
            wimg0=img0
            wimg1=img1
            flow = torch.ones_like(a[:,:4],device=a.device)*1e-6
            mask = torch.ones_like(a[:,:1],device=a.device)*0.5

        x = torch.cat((a,img0,img1,wimg0,wimg1,flow,mask,t),dim=1)
        x_ = self.bodyconv(x)
        x =  self.flowconv(x_)
        flow = x[:,:4]  + flow
        mask = x[:,4:5] + mask 


        if res == None:
            res = self.resconv(torch.cat( (backward_warp(stem[2][:B],flow[:,:2]),
                                           backward_warp(stem[2][B:],flow[:,2:4]),
                                           flow,
                                           mask,
                                           x_),1))
        else:
            res = self.resconv(torch.cat((
                                          backward_warp(stem[1][:B],flow[:,:2]),
                                          backward_warp(stem[1][B:],flow[:,2:4]),
                                          flow,
                                          mask,
                                          x_,res),1))
    
        if self.scale >1.0:
            if self.scale != 4:
                flow_up=F.interpolate(flow,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)*(self.scale // 4)
                mask_up=F.interpolate(mask,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)

        if self.last:
            res = self.lastconv(torch.cat((
                                           flow_up,
                                           mask_up,
                                           res),1))


        return flow_up,mask_up,flow, mask,res
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################      
class Head_pme_ca2_2(nn.Module):
    def __init__(self, in_planes, scale,c ,last):
        super(Head_pme_ca2_2, self).__init__()

        self.scale = scale
        self.last= last
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PixelShuffle(2))

        #self.first_conv = conv(in_planes*2,c,3,1,1)
        self.ca = ChannelAttention(in_planes*2,16)
        self.bodyconv = nn.Sequential(
            conv(in_planes*2//16 +12+5+1,c, 3, 1,1),
            conv(c, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
                                  ) 
        self.flowconv = nn.Sequential(
            conv(c, 5, 3, 1, 1))
        
        if self.last:
            self.resconv = nn.Sequential(Conv2(c+c//2+c + 5,c,3,1,1),deconv(c,c//2))  # 2c *2 + res_i-1 + c
            self.lastconv= nn.Sequential(Conv2(c//2 + 5 ,c,3,1,1),nn.Conv2d(c,3,3,1,1)) #4c //2 =2c
        else:
            self.resconv = nn.Sequential(Conv2(2*c+c + 5 ,c,3,1,1),deconv(c,c//2)) 
    
    def forward(self,a0,a1,stem,res,flow,mask,img0,img1,timestep): # /16 /8 /4
        B,C,H,W=a0.shape
        if self.scale > 1.0:
            a= self.upsample(self.ca(torch.cat((a0,a1),1)))

        o_img0= img0
        o_img1= img1

        if img0.shape[-1] > a0.shape[-1]:
            img0=F.interpolate(img0,size= a.shape[-2:],mode="bilinear", align_corners=False)
            img1=F.interpolate(img1,size= a.shape[-2:],mode="bilinear", align_corners=False)
        
        t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
        
        if flow != None:
            flow = F.interpolate(flow,scale_factor=2.0,mode="bilinear",align_corners=False)*(2.0)
            mask = F.interpolate(mask,scale_factor=2.0,mode="bilinear",align_corners=False)
            wimg0    =  backward_warp(img0,flow[:,0:2])
            wimg1    =  backward_warp(img1,flow[:,2:4])
        
        if flow == None:
            wimg0=img0
            wimg1=img1
            flow = torch.ones_like(a[:,:4],device=a.device)*1e-6
            mask = torch.ones_like(a[:,:1],device=a.device)*0.5

        x = torch.cat((a,img0,img1,wimg0,wimg1,flow,mask,t),dim=1)
        x_ = self.bodyconv(x)
        x =  self.flowconv(x_)
        flow = x[:,:4]  + flow
        mask = x[:,4:5] + mask 


        if res == None:
            res = self.resconv(torch.cat( (backward_warp(stem[2][:B],flow[:,:2]),
                                           backward_warp(stem[2][B:],flow[:,2:4]),
                                           flow,
                                           mask,
                                           x_),1))
        else:
            res = self.resconv(torch.cat((
                                          backward_warp(stem[1][:B],flow[:,:2]),
                                          backward_warp(stem[1][B:],flow[:,2:4]),
                                          flow,
                                          mask,
                                          x_,res),1))
    
        if self.scale >1.0:
            if self.scale != 4:
                flow_up=F.interpolate(flow,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)*(self.scale // 4)
                mask_up=F.interpolate(mask,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)

        if self.last:
            res = self.lastconv(torch.cat((
                                           backward_warp(o_img0,flow_up[:,0:2]),
                                           backward_warp(o_img1,flow_up[:,2:4]),
                                           o_img0,
                                           o_img1,
                                           flow_up,
                                           mask_up,
                                           res),1))


        return flow_up,mask_up,flow, mask,res
    
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################      
class Head_pme_ca3(nn.Module):
    def __init__(self, in_planes, scale,c ,last):
        super(Head_pme_ca3, self).__init__()

        self.scale = scale
        self.last= last
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PixelShuffle(2))

        #self.first_conv = conv(in_planes*2,c,3,1,1)
        self.ca = ChannelAttention(in_planes*2,16)
        self.bodyconv = nn.Sequential(
            conv(in_planes*2//16 +12+5+1,c, 3, 1,1),
            conv(c, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
                                  ) 
        self.flowconv = nn.Sequential(
            conv(c, 5, 3, 1, 1))
        
        if self.last:
            self.resconv = nn.Sequential(Conv2(c+c//2+c + 17,c,3,1,1),deconv(c,c//2))  # 2c *2 + res_i-1 + c
            self.lastconv= nn.Sequential(Conv2(c//2+c//2 + 17 ,c,3,1,1),nn.Conv2d(c,3,3,1,1)) #4c //2 =2c
        else:
            self.resconv = nn.Sequential(Conv2(2*c+c + 17 ,c,3,1,1),deconv(c,c//2)) 
    
    def forward(self,a0,a1,stem,res,flow,mask,img0,img1,timestep): # /16 /8 /4
        B,C,H,W=a0.shape
        if self.scale > 1.0:
            a= self.upsample(self.ca(torch.cat((a0,a1),1)))

        o_img0=img0
        o_img1=img1
        if img0.shape[-1] > a0.shape[-1]:
            img0=F.interpolate(img0,size= a.shape[-2:],mode="bilinear", align_corners=False)
            img1=F.interpolate(img1,size= a.shape[-2:],mode="bilinear", align_corners=False)
        
        t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
        
        if flow != None:
            flow = F.interpolate(flow,scale_factor=2.0,mode="bilinear",align_corners=False)*(2.0)
            mask = F.interpolate(mask,scale_factor=2.0,mode="bilinear",align_corners=False)
            wimg0    =  backward_warp(img0,flow[:,0:2])
            wimg1    =  backward_warp(img1,flow[:,2:4])
        
        if flow == None:
            wimg0=img0
            wimg1=img1
            flow = torch.ones_like(a[:,:4],device=a.device)*1e-6
            mask = torch.ones_like(a[:,:1],device=a.device)*0.5

        x = torch.cat((a,img0,img1,wimg0,wimg1,flow,mask,t),dim=1)
        x_ = self.bodyconv(x)
        x =  self.flowconv(x_)
        flow = x[:,:4]  + flow
        mask = x[:,4:5] + mask 


        if res == None:
            res = self.resconv(torch.cat( (backward_warp(stem[2][:B],flow[:,:2]),
                                           backward_warp(stem[2][B:],flow[:,2:4]),
                                           backward_warp(img0,flow[:,0:2]),
                                           backward_warp(img1,flow[:,2:4]),
                                           img0,
                                           img1,
                                           flow,
                                           mask,
                                           x_),1))
        else:
            res = self.resconv(torch.cat((
                                          backward_warp(stem[1][:B],flow[:,:2]),
                                          backward_warp(stem[1][B:],flow[:,2:4]),
                                           backward_warp(img0,flow[:,0:2]),
                                           backward_warp(img1,flow[:,2:4]),
                                           img0,
                                           img1,
                                           flow,
                                           mask,
                                           x_,res),1))
    
        if self.scale >1.0:
            if self.scale != 4:
                flow_up=F.interpolate(flow,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)*(self.scale // 4)
                mask_up=F.interpolate(mask,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)

        if self.last:
            res = self.lastconv(torch.cat((backward_warp(stem[0][:B],flow_up[:,:2] ),
                                           backward_warp(stem[0][B:],flow_up[:,2:4]),
                                            backward_warp(o_img0,flow_up[:,0:2]),
                                           backward_warp(o_img1,flow_up[:,2:4]),
                                           o_img0,
                                           o_img1,
                                           flow_up,
                                           mask_up,
                                           res),1))


        return flow_up,mask_up,flow, mask,res

###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################      
class Head_pme_ca4(nn.Module):
    def __init__(self, in_planes, scale,c ,last):
        super(Head_pme_ca4, self).__init__()

        self.scale = scale
        self.last= last
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PixelShuffle(2))

        #self.first_conv = conv(in_planes*2,c,3,1,1)
        self.ca = ChannelAttention(in_planes*2,16)
        self.bodyconv = nn.Sequential(
            #mlp(in_planes*2//16 +12+5+1,4*c,c, 3, 1,1),
            conv(in_planes*2//16 +12+5+1,c, 3, 1,1),
            conv(c, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
            #conv(c, c, 3, 1, 1),
                                  ) 
        self.flowconv = nn.Sequential(
            conv(c, 5, 3, 1, 1))
        
        if self.last == False :
            self.resconv = nn.Sequential(mlp(2*c+c+5,4*c,c ,3,1,1),RB(c,c),RB(c,c), deconv(c,c//2)) 
        else:
            self.resconv = nn.Sequential(mlp(c+c//2+c + 5, 2*c, c//2,3,1,1),RB(c//2,c//2),RB(c//2,c//2), deconv(c//2,c//4))  # 2c *2 + res_i-1 + c
            self.lastconv= nn.Sequential(mlp(c//4+c//2 + 5 ,2*c//2, c//4 ,3,1,1),RB(c//4,c//4),RB(c//4,c//4), nn.Conv2d(c//4,3,3,1,1)) #4c //2 =2c
    
    def forward(self,a0,a1,stem,res,flow,mask,img0,img1,timestep): # /16 /8 /4
        B,C,H,W=a0.shape
        if self.scale > 1.0:
            a= self.upsample(self.ca(torch.cat((a0,a1),1)))


        if img0.shape[-1] > a0.shape[-1]:
            img0=F.interpolate(img0,size= a.shape[-2:],mode="bilinear", align_corners=False)
            img1=F.interpolate(img1,size= a.shape[-2:],mode="bilinear", align_corners=False)
        
        t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
        
        if flow != None:
            flow = F.interpolate(flow,scale_factor=2.0,mode="bilinear",align_corners=False)*(2.0)
            mask = F.interpolate(mask,scale_factor=2.0,mode="bilinear",align_corners=False)
            wimg0    =  backward_warp(img0,flow[:,0:2])
            wimg1    =  backward_warp(img1,flow[:,2:4])
        
        if flow == None:
            wimg0=img0
            wimg1=img1
            flow = torch.ones_like(a[:,:4],device=a.device)*1e-6
            mask = torch.ones_like(a[:,:1],device=a.device)*0.5

        x = torch.cat((a,img0,img1,wimg0,wimg1,flow,mask,t),dim=1)
        x_ = self.bodyconv(x)
        x =  self.flowconv(x_)
        flow = x[:,:4]  + flow
        mask = x[:,4:5] + mask 


        if res == None:
            res = self.resconv(torch.cat( (backward_warp(stem[2][:B],flow[:,:2]),
                                           backward_warp(stem[2][B:],flow[:,2:4]),
                                           flow,
                                           mask,
                                           x_),1))
        else:
            res = self.resconv(torch.cat((
                                          backward_warp(stem[1][:B],flow[:,:2]),
                                          backward_warp(stem[1][B:],flow[:,2:4]),
                                          flow,
                                          mask,
                                          x_,res),1))
    
        if self.scale >1.0:
            if self.scale != 4:
                flow_up=F.interpolate(flow,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)*(self.scale // 4)
                mask_up=F.interpolate(mask,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)

        if self.last:
            res = self.lastconv(torch.cat((backward_warp(stem[0][:B],flow_up[:,:2]),
                                           backward_warp(stem[0][B:],flow_up[:,2:4]),
                                           flow_up,
                                           mask_up,
                                           res),1))


        return flow_up,mask_up,flow, mask,res
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################  
###########################################################################################################################################################################      
class Head_pme_cabn(nn.Module):
    def __init__(self, in_planes, scale,c ,last):
        super(Head_pme_cabn, self).__init__()

        self.scale = scale
        self.last= last
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PixelShuffle(2))

        #self.first_conv = conv(in_planes*2,c,3,1,1)
        self.ca = ChannelAttention(in_planes*2,16)
        self.bodyconv = nn.Sequential(
            convbn(in_planes*2//16 +12+5+1,c, 3, 1,1),
            convbn(c, c, 3, 1, 1),
            convbn(c, c, 3, 1, 1),
                                  ) 
        self.flowconv = nn.Sequential(
            conv(c, 5, 3, 1, 1))
        
        if self.last:
            self.resconv = nn.Sequential(Conv2bn(c+c//2+c,c,3,1,1),deconvbn(c,c//2))  # 2c *2 + res_i-1 + c
            self.lastconv= nn.Sequential(Conv2bn(c//2+c//2,c,3,1,1),nn.Conv2d(c,3,3,1,1)) #4c //2 =2c
        else:
            self.resconv = nn.Sequential(Conv2bn(2*c+c,c,3,1,1),deconvbn(c,c//2)) 
    
    def forward(self,a0,a1,stem,res,flow,mask,img0,img1,timestep): # /16 /8 /4
        B,C,H,W=a0.shape
        if self.scale > 1.0:
            a= self.upsample(self.ca(torch.cat((a0,a1),1)))


        if img0.shape[-1] > a0.shape[-1]:
            img0=F.interpolate(img0,size= a.shape[-2:],mode="bilinear", align_corners=False)
            img1=F.interpolate(img1,size= a.shape[-2:],mode="bilinear", align_corners=False)
        
        t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
        
        if flow != None:
            flow = F.interpolate(flow,scale_factor=2.0,mode="bilinear",align_corners=False)*(2.0)
            mask = F.interpolate(mask,scale_factor=2.0,mode="bilinear",align_corners=False)
            wimg0    =  backward_warp(img0,flow[:,0:2])
            wimg1    =  backward_warp(img1,flow[:,2:4])
        
        if flow == None:
            wimg0=img0
            wimg1=img1
            flow = torch.ones_like(a[:,:4],device=a.device)*1e-6
            mask = torch.ones_like(a[:,:1],device=a.device)*0.5

        x = torch.cat((a,img0,img1,wimg0,wimg1,flow,mask,t),dim=1)
        x_ = self.bodyconv(x)
        x =  self.flowconv(x_)
        flow = x[:,:4]  + flow
        mask = x[:,4:5] + mask 


        if res == None:
            res = self.resconv(torch.cat( (backward_warp(stem[2][:B],flow[:,:2]),
                                           backward_warp(stem[2][B:],flow[:,2:4]),
                                           x_),1))
        else:
            res = self.resconv(torch.cat((
                                          backward_warp(stem[1][:B],flow[:,:2]),
                                          backward_warp(stem[1][B:],flow[:,2:4]),
                                          x_,res),1))
    
        if self.scale >1.0:
            if self.scale != 4:
                flow_up=F.interpolate(flow,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)*(self.scale // 4)
                mask_up=F.interpolate(mask,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)

        if self.last:
            res = self.lastconv(torch.cat((backward_warp(stem[0][:B],flow_up[:,:2]),
                                           backward_warp(stem[0][B:],flow_up[:,2:4])
                                           ,res),1))


        return flow_up,mask_up,flow, mask,res



###########################################################################################################################################################################

class Head_pme_ca_fast(nn.Module):
    def __init__(self, in_planes, scale,c ,last,is_inference=False):
        super(Head_pme_ca_fast, self).__init__()

        self.scale = scale
        self.last= last
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PixelShuffle(2))

        #self.first_conv = conv(in_planes*2,c,3,1,1)
        self.ca = ChannelAttention(in_planes*2,16)
        '''self.bodyconv = nn.Sequential(
            conv(in_planes*2//16 +12+5+1,c, 3, 1,1),
            conv(c, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
                                  ) '''
        self.bodyconv = nn.Sequential(
            MobileOneBlock(
                    in_channels=in_planes*2//16 +12+5+1,
                    out_channels=c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=1,
                    inference_mode=is_inference,
                    use_se=False,
                    num_conv_branches=1,),
 
            MobileOneBlock(
                    in_channels=c,
                    out_channels=c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=1,
                    inference_mode=is_inference,
                    use_se=False,
                    num_conv_branches=1,),
 
            MobileOneBlock(
                    in_channels=c,
                    out_channels=c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=1,
                    inference_mode=is_inference,
                    use_se=False,
                    num_conv_branches=1,)
 
                                  ) 
        self.flowconv = nn.Sequential(
                  MobileOneBlock(
                    in_channels=c,
                    out_channels=5,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=1,
                    inference_mode=is_inference,
                    use_se=False,
                    num_conv_branches=1,))
        
        if self.last:
            self.resconv = nn.Sequential(
                                        MobileOneBlock(
                                        in_channels=c+c//2+c,
                                        out_channels=c,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        inference_mode=is_inference,
                                        use_se=False,
                                        num_conv_branches=1,),
                                         
                                         deconv(c,c//2))  # 2c *2 + res_i-1 + c
            self.lastconv= nn.Sequential(
                                        MobileOneBlock(
                                        in_channels=c//2+c//2,
                                        out_channels=c,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        inference_mode=is_inference,
                                        use_se=False,
                                        num_conv_branches=1,),
                                         
                                         nn.Conv2d(c,3,3,1,1)) #4c //2 =2c
        else:
            self.resconv = nn.Sequential(
                                        MobileOneBlock(
                                        in_channels=2*c+c,
                                        out_channels=c,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=1,
                                        inference_mode=is_inference,
                                        use_se=False,
                                        num_conv_branches=1,),
                                         deconv(c,c//2)) 
    
    def forward(self,a0,a1,stem,res,flow,mask,img0,img1,timestep): # /16 /8 /4
        B,C,H,W=a0.shape
        if self.scale > 1.0:
            a= self.upsample(self.ca(torch.cat((a0,a1),1)))


        if img0.shape[-1] > a0.shape[-1]:
            img0=F.interpolate(img0,size= a.shape[-2:],mode="bilinear", align_corners=False)
            img1=F.interpolate(img1,size= a.shape[-2:],mode="bilinear", align_corners=False)
        
        t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
        
        if flow != None:
            flow = F.interpolate(flow,scale_factor=2.0,mode="bilinear",align_corners=False)*(2.0)
            mask = F.interpolate(mask,scale_factor=2.0,mode="bilinear",align_corners=False)
            wimg0    =  backward_warp(img0,flow[:,0:2])
            wimg1    =  backward_warp(img1,flow[:,2:4])
        
        if flow == None:
            wimg0=img0
            wimg1=img1
            flow = torch.ones_like(a[:,:4],device=a.device)*1e-6
            mask = torch.ones_like(a[:,:1],device=a.device)*0.5

        x = torch.cat((a,img0,img1,wimg0,wimg1,flow,mask,t),dim=1)
        x_ = self.bodyconv(x)
        x =  self.flowconv(x_)
        flow = x[:,:4]  + flow
        mask = x[:,4:5] + mask 


        if res == None:
            res = self.resconv(torch.cat( (backward_warp(stem[2][:B],flow[:,:2]),
                                           backward_warp(stem[2][B:],flow[:,2:4]),
                                           x_),1))
        else:
            res = self.resconv(torch.cat((
                                          backward_warp(stem[1][:B],flow[:,:2]),
                                          backward_warp(stem[1][B:],flow[:,2:4]),
                                          x_,res),1))
    
        if self.scale >1.0:
            if self.scale != 4:
                flow_up=F.interpolate(flow,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)*(self.scale // 4)
                mask_up=F.interpolate(mask,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)

        if self.last:
            res = self.lastconv(torch.cat((backward_warp(stem[0][:B],flow_up[:,:2]),
                                           backward_warp(stem[0][B:],flow_up[:,2:4])
                                           ,res),1))


        return flow_up,mask_up,flow, mask,res



class Head_pme_RB(nn.Module):
    def __init__(self, in_planes, scale,c ,last):
        super(Head_pme_RB, self).__init__()

        self.scale = scale
        self.last= last
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PixelShuffle(2))
        self.ca = ChannelAttention(in_planes*2,16)

        self.bodyconv = nn.Sequential(
            conv(in_planes*2//16 +12+5+1, c, 3, 1,1),
            conv(c, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
                                  ) 
        self.flowconv = nn.Sequential(
            conv(c, 5, 3, 1, 1))
        
        if self.last:
            self.resconv = nn.Sequential(conv(c+c//2+c,c,3,1,1),RB(c,c,3,1,1),RB(c,c,3,1,1),deconv(c,c//2))  # 2c *2 + res_i-1 + c
            self.lastconv= nn.Sequential(conv(c//2+c//2,c,3,1,1),nn.Conv2d(c,3,3,1,1)) #4c //2 =2c
        else:
            self.resconv = nn.Sequential(conv(2*c+c,c,3,1,1),RB(c,c,3,1,1),RB(c,c,3,1,1),deconv(c,c//2)) 
    
    def forward(self,a0,a1,stem,res,flow,mask,img0,img1,timestep): # /16 /8 /4
        B,C,H,W=a0.shape
        if self.scale > 1.0:
            a= self.upsample(self.ca(torch.cat((a0,a1),1)))
  
        if img0.shape[-1] > a0.shape[-1]:
            img0=F.interpolate(img0,size= a.shape[-2:],mode="bilinear", align_corners=False)
            img1=F.interpolate(img1,size= a.shape[-2:],mode="bilinear", align_corners=False)
        
        t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
        
        if flow != None:
            flow = F.interpolate(flow,scale_factor=2.0,mode="bilinear",align_corners=False)*(2.0)
            mask = F.interpolate(mask,scale_factor=2.0,mode="bilinear",align_corners=False)
            wimg0    =  backward_warp(img0,flow[:,0:2])
            wimg1    =  backward_warp(img1,flow[:,2:4])
        
        if flow == None:
            wimg0=img0
            wimg1=img1
            flow = torch.ones_like(a[:,:4],device=a.device)*1e-6
            mask = torch.ones_like(a[:,:1],device=a.device)*0.5

        x = torch.cat((a,img0,img1,wimg0,wimg1,flow,mask,t),dim=1)
        x_ = self.bodyconv(x)
        x =  self.flowconv(x_)
        flow = x[:,:4]  + flow
        mask = x[:,4:5] + mask 


        if res == None:
            res = self.resconv(torch.cat( (backward_warp(stem[2][:B],flow[:,:2]),
                                           backward_warp(stem[2][B:],flow[:,2:4]),
                                           x_),1))
        else:
            res = self.resconv(torch.cat((
                                          backward_warp(stem[1][:B],flow[:,:2]),
                                          backward_warp(stem[1][B:],flow[:,2:4]),
                                          x_,res),1))
    
        if self.scale >1.0:
            if self.scale != 4:
                flow_up=F.interpolate(flow,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)*(self.scale // 4)
                mask_up=F.interpolate(mask,scale_factor=self.scale // 4,mode="bilinear",align_corners=False)

        if self.last:
            res = self.lastconv(torch.cat((backward_warp(stem[0][:B],flow_up[:,:2]),
                                           backward_warp(stem[0][B:],flow_up[:,2:4])
                                           ,res),1))


        return flow_up,mask_up,flow, mask,res




class Head_ms_ps(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head_ms_ps, self).__init__()
        self.upsample0 = nn.Identity()
        self.upsample1 = nn.Sequential(nn.PixelShuffle(2))
        self.upsample2 = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale

        self.conv0 = nn.Sequential(
                                  conv(in_planes*2  + in_else, 4*c),
                                  conv(4*c, 4*c),
                                  conv(4*c, 5),
                                  )  
        self.conv1 = nn.Sequential(
                                  conv(in_planes*2 // (4) + in_else  + 5, 2*c),
                                  conv(2*c, 2*c),
                                  conv(2*c, 5),
                                  )  
        self.conv2 = nn.Sequential(
                                  conv(in_planes*2 // (4*4) + in_else + 5, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  

    def forward(self, motion_feature, x, flow): # /16 /8 /4
        motion_feature0 = self.upsample0(motion_feature) #/4 /2 /1
        motion_feature1 = self.upsample1(motion_feature) #/4 /2 /1
        motion_feature2 = self.upsample2(motion_feature) #/4 /2 /1

        if self.scale != 4:
            x2 = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
            x1 = F.interpolate(x, scale_factor = 2. / self.scale, mode="bilinear", align_corners=False)
            x0 = F.interpolate(x, scale_factor = 1. / self.scale, mode="bilinear", align_corners=False)

        if flow != None:
            if self.scale != 4:
                flow0 = F.interpolate(flow, scale_factor = 1. / self.scale, mode="bilinear", align_corners=False) * 1. / self.scale
                flow1 = F.interpolate(flow, scale_factor = 2. / self.scale, mode="bilinear", align_corners=False) * 2. / self.scale
                flow2 = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
   
            x0 = torch.cat((x0, flow0), 1)
            x1 = torch.cat((x1, flow1), 1)
            x2 = torch.cat((x2, flow2), 1)

        x0 = self.conv0(torch.cat([motion_feature0, x0], 1))
        flow0 = x0[:,:4]
        mask0 = x0[:,4:5]
        flow0=F.interpolate(flow0, scale_factor=2.0, mode='bilinear',align_corners=False)*2.0
        mask0=F.interpolate(mask0, scale_factor=2.0, mode='bilinear',align_corners=False)


        x1 = self.conv1(torch.cat([motion_feature1,x1,flow0,mask0],1))
        flow1 = x1[:,:4] + flow0
        mask1 = x1[:,4:5] +mask0
        flow1=F.interpolate(flow1, scale_factor=2.0, mode='bilinear',align_corners=False)*2.0
        mask1=F.interpolate(mask1, scale_factor=2.0, mode='bilinear',align_corners=False)


        x2 = self.conv2(torch.cat([motion_feature2,x2,flow1,mask1],1))
        
        flow2 = x2[:,:4] + flow1
        mask2 = x2[:,4:5] +mask1
        

        if self.scale != 4:
            flow = F.interpolate(flow2, scale_factor = self.scale / 4, mode="bilinear", align_corners=False) * (self.scale // 4)
            mask = F.interpolate(mask2,scale_factor=self.scale/4 ,mode='bilinear', align_corners=False)
        else:
            flow = flow2
            mask = mask2

        return flow, mask


class Head_ms_ps2(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head_ms_ps2, self).__init__()
        self.upsample0 = nn.Identity()
        self.upsample1 = nn.Sequential(nn.PixelShuffle(2))
        self.upsample2 = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.upsample3 = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale

        self.conv0 = nn.Sequential(
                                  conv(in_planes*2  + in_else, 4*c), #512 - 256
                                  conv(4*c, 4*c),
                                  conv(4*c, 5),
                                  )  
        self.conv1 = nn.Sequential(
                                  conv(in_planes*2 // (4) + in_else  + 5, 2*c), #128 - 128
                                  conv(2*c, 2*c),
                                  conv(2*c, 5),
                                  )  
        self.conv2 = nn.Sequential(
                                  conv(in_planes*2 // (4*4) + in_else + 5, c), #32  - 64
                                  conv(c, c),
                                  conv(c, 5),
                                  )  
        self.conv3 = nn.Sequential(
                                  conv(in_planes*2 // (4*4*4) + in_else + 5, c//2), #8 - 32
                                  conv(c//2, c//2),
                                  conv(c//2, 5),
                                  )  


    def forward(self, motion_feature, x, flow): # /16 /8 /4
        motion_feature0 = self.upsample0(motion_feature) #/4 /2 /1 32 
        motion_feature1 = self.upsample1(motion_feature) #/4 /2 /1 64 
        motion_feature2 = self.upsample2(motion_feature) #/4 /2 /1 128
        motion_feature3 = self.upsample3(motion_feature) #/4 /2 /1 256

        if self.scale != 4:
            x3 = F.interpolate(x, scale_factor = 8. / self.scale, mode="bilinear", align_corners=False)
            x2 = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
            x1 = F.interpolate(x, scale_factor = 2. / self.scale, mode="bilinear", align_corners=False)
            x0 = F.interpolate(x, scale_factor = 1. / self.scale, mode="bilinear", align_corners=False)

        if flow != None:
            if self.scale != 4:
                flow0 = F.interpolate(flow, scale_factor = 1. / self.scale, mode="bilinear", align_corners=False) * 1. / self.scale
                flow1 = F.interpolate(flow, scale_factor = 2. / self.scale, mode="bilinear", align_corners=False) * 2. / self.scale
                flow2 = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
                flow3 = F.interpolate(flow, scale_factor = 8. / self.scale, mode="bilinear", align_corners=False) * 8. / self.scale
   
            x0 = torch.cat((x0, flow0), 1)
            x1 = torch.cat((x1, flow1), 1)
            x2 = torch.cat((x2, flow2), 1)
            x3 = torch.cat((x3, flow3), 1)

        x0 = self.conv0(torch.cat([motion_feature0, x0], 1))
        flow0 = x0[:,:4]
        mask0 = x0[:,4:5]
        flow0=F.interpolate(flow0, scale_factor=2.0, mode='bilinear',align_corners=False)*2.0
        mask0=F.interpolate(mask0, scale_factor=2.0, mode='bilinear',align_corners=False)


        x1 = self.conv1(torch.cat([motion_feature1,x1,flow0,mask0],1))
        flow1 = x1[:,:4] + flow0
        mask1 = x1[:,4:5] +mask0
        flow1=F.interpolate(flow1, scale_factor=2.0, mode='bilinear',align_corners=False)*2.0
        mask1=F.interpolate(mask1, scale_factor=2.0, mode='bilinear',align_corners=False)

        x2 = self.conv2(torch.cat([motion_feature2,x2,flow1,mask1],1))
        flow2 = x2[:,:4] + flow1
        mask2 = x2[:,4:5] +mask1
        flow2=F.interpolate(flow2, scale_factor=2.0, mode='bilinear',align_corners=False)*2.0
        mask2=F.interpolate(mask2, scale_factor=2.0, mode='bilinear',align_corners=False)


        x3 = self.conv3(torch.cat([motion_feature3,x3,flow2,mask2],1))
        flow3 = x3[:,:4] + flow2
        mask3 = x3[:,4:5] +mask2
        

        if self.scale != 8:
            flow = F.interpolate(flow3, scale_factor = self.scale / 8, mode="bilinear", align_corners=False) * (self.scale // 8)
            mask = F.interpolate(mask3,scale_factor=self.scale/8 ,mode='bilinear', align_corners=False)
        else:
            flow = flow3
            mask = mask3

        return flow, mask
    

class Head_ms_ps3(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head_ms_ps3, self).__init__()
    
        self.upsample2 = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.upsample3 = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale

       
        self.conv2 = nn.Sequential(
                                  conv(in_planes*2 // (4*4) + in_else , c), #32  - 64
                                  conv(c, c),
                                  conv(c, 5),
                                  )  
        self.conv3 = nn.Sequential(
                                  conv(in_planes*2 // (4*4*4) + in_else + 5, c//2), #8 - 32
                                  conv(c//2, c//2),
                                  conv(c//2, 5),
                                  )  


    def forward(self, motion_feature, x, flow): # /16 /8 /4

        motion_feature2 = self.upsample2(motion_feature) #/4 /2 /1 128
        motion_feature3 = self.upsample3(motion_feature) #/4 /2 /1 256

        if self.scale != 4:
            x2 = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
            x3 = F.interpolate(x, scale_factor = 8. / self.scale, mode="bilinear", align_corners=False)


        if flow != None:
            if self.scale != 4:
                flow2 = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
                flow3 = F.interpolate(flow, scale_factor = 8. / self.scale, mode="bilinear", align_corners=False) * 8. / self.scale
   

            x2 = torch.cat((x2, flow2), 1)
            x3 = torch.cat((x3, flow3), 1)


        x2 = self.conv2(torch.cat([motion_feature2,x2],1))
        flow2 = x2[:,:4] 
        mask2 = x2[:,4:5]
        flow2=F.interpolate(flow2, scale_factor=2.0, mode='bilinear',align_corners=False)*2.0
        mask2=F.interpolate(mask2, scale_factor=2.0, mode='bilinear',align_corners=False)


        x3 = self.conv3(torch.cat([motion_feature3,x3,flow2,mask2],1))
        flow3 = x3[:,:4] + flow2
        mask3 = x3[:,4:5] +mask2
        

        if self.scale != 8:
            flow = F.interpolate(flow3, scale_factor = self.scale / 8, mode="bilinear", align_corners=False) * (self.scale // 8)
            mask = F.interpolate(mask3,scale_factor=self.scale/8 ,mode='bilinear', align_corners=False)
        else:
            flow = flow3
            mask = mask3

        return flow, mask
    

class Head_ms_bi(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head_ms_bi, self).__init__()
        self.upsample0 = nn.Identity()
    
        self.scale = scale

        self.conv0 = nn.Sequential(
                                  conv(in_planes*2  + in_else, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  
        self.conv1 = nn.Sequential(
                                  conv(in_planes*2  + in_else  + 5, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  
        self.conv2 = nn.Sequential(
                                  conv(in_planes*2  + in_else + 5, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  

    def forward(self, motion_feature, x, flow): # /16 /8 /4
        motion_feature0 = self.upsample0(motion_feature) #/4 /2 /1
        motion_feature1 = F.interpolate(motion_feature, scale_factor = 2. , mode="bilinear", align_corners=False)
        motion_feature2 = F.interpolate(motion_feature, scale_factor = 4. , mode="bilinear", align_corners=False)

        if self.scale != 4:
            x2 = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
            x1 = F.interpolate(x, scale_factor = 2. / self.scale, mode="bilinear", align_corners=False)
            x0 = F.interpolate(x, scale_factor = 1. / self.scale, mode="bilinear", align_corners=False)

        if flow != None:
            if self.scale != 4:
                flow0 = F.interpolate(flow, scale_factor = 1. / self.scale, mode="bilinear", align_corners=False) * 1. / self.scale
                flow1 = F.interpolate(flow, scale_factor = 2. / self.scale, mode="bilinear", align_corners=False) * 2. / self.scale
                flow2 = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
   
            x0 = torch.cat((x0, flow0), 1)
            x1 = torch.cat((x1, flow1), 1)
            x2 = torch.cat((x2, flow2), 1)

        x0 = self.conv0(torch.cat([motion_feature0, x0], 1))
        flow0 = x0[:,:4]
        mask0 = x0[:,4:5]
        flow0=F.interpolate(flow0, scale_factor=2.0, mode='bilinear',align_corners=False)*2.0
        mask0=F.interpolate(mask0, scale_factor=2.0, mode='bilinear',align_corners=False)


        x1 = self.conv1(torch.cat([motion_feature1,x1,flow0,mask0],1))
        flow1 = x1[:,:4] + flow0
        mask1 = x1[:,4:5] +mask0
        flow1=F.interpolate(flow1, scale_factor=2.0, mode='bilinear',align_corners=False)*2.0
        mask1=F.interpolate(mask1, scale_factor=2.0, mode='bilinear',align_corners=False)


        x2 = self.conv2(torch.cat([motion_feature2,x2,flow1,mask1],1))
        
        flow2 = x2[:,:4] + flow1
        mask2 = x2[:,4:5] +mask1
        

        if self.scale != 4:
            flow = F.interpolate(flow2, scale_factor = self.scale / 4, mode="bilinear", align_corners=False) * (self.scale // 4)
            mask = F.interpolate(mask2,scale_factor=self.scale/4 ,mode='bilinear', align_corners=False)
        else:
            flow = flow2
            mask = mask2

        return flow, mask
    



