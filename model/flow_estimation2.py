import torch
import torch.nn as nn
import torch.nn.functional as F
from .refine import *
from .layers2 import * 
from .visualize import * 
from .geometry import *
import wandb as wandb
from .feature_extractor import * 
from .warplayer import * 

    
###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################  

class MultiScaleFlow_dec_opt(nn.Module):
    def __init__(self, backbone, **kargs):
        super().__init__()
        print('MultiScaleFlow_dec_opt')
        self.refine= kargs['refine']
        self.atime=kargs['atime']
        self.motion=kargs['motion']
        self.flow_num_stage = len(kargs['hidden_dims'])
        self.feature_bone = backbone

        if self.motion :
            self.block = nn.ModuleList([Head( kargs['motion_dims'][-1-i] * kargs['depths'][-1-i] + kargs['embed_dims'][-1-i], 
                                kargs['scales'][-1-i], 
                                kargs['hidden_dims'][-1-i],
                                6 if i==0 else 17 )
                                for i in range(self.flow_num_stage)])
        else :
            self.block = nn.ModuleList([Head( kargs['embed_dims'][-1-i], 
                                kargs['scales'][-1-i], 
                                kargs['hidden_dims'][-1-i],
                                7 if i==0 else 18 )
                                for i in range(self.flow_num_stage)])
        
        if self.refine != None :
            self.unet = (kargs['refine'](int(kargs['c']*2)))

    def warp_features(self, xs, flows):
        y0 = []
        B = xs[0].size(0) // 2

        for i,x in enumerate(xs):
            #print(x.shape ,flows[i].shape)
            y0.append(
                torch.cat( (
                backward_warp(x[:B], flows[i][:, 0:2]),
                backward_warp(x[B:], flows[i][:, 2:4])),1))
            #print(y0[-1].shape)
        return y0
    
    def warp_imgs(self, imgs, flows):
        y = []
        
        for (img,flow) in (zip(imgs,flows)) :
            y.append(
                torch.cat((
                backward_warp(img[:,:3], flow[ :,0:2]),
                backward_warp(img[:,3:6], flow[ :,2:4])),1))
            
        return y

    
    def pyramid_features(self, x,level=[1,0.5,0.25],scale=False):
        xs=[]
        for lv in level :
            if lv != 1.0 :
                if scale :
                    xs.append( F.interpolate(x, scale_factor=lv, mode="bilinear", align_corners=False, recompute_scale_factor=False) * lv)                 
                else:
                    xs.append( F.interpolate(x, scale_factor=lv, mode="bilinear", align_corners=False, recompute_scale_factor=False))
            else:
                xs.append(x)
        return xs
    
    def u_pyramid_features(self,img,flow,mask,level=[1,0.5,0.25] ):

        im,f,m=[],[],[]
        for lv in level :
            if lv != 1.0 :
                    im.append( F.interpolate(img, scale_factor=lv, mode="bilinear", align_corners=False, recompute_scale_factor=False) )     
                    f.append( F.interpolate(flow, scale_factor=lv, mode="bilinear", align_corners=False, recompute_scale_factor=False) * lv)     
                    m.append( F.interpolate(mask, scale_factor=lv, mode="bilinear", align_corners=False, recompute_scale_factor=False) )     

            else:
                    im.append(img)     
                    f.append(flow)     
                    m.append(mask)     

        return im,f,m
    
    
    def calculate_flow(self, imgs, timestep, af=None, mf=None):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None
        # appearence_features & motion_features
        if self.motion:
            if (af is None) or (mf is None):
                af, mf = self.feature_bone(img0, img1)
        else :
            if (af is None) :
                af = self.feature_bone(img0, img1)

        for i in range(self.flow_num_stage):
            #t = torch.full(mf[-1-i][:B].shape, timestep, dtype=torch.float).cuda()

            if flow != None:
                warped_img0 = backward_warp(img0, flow[:, :2])
                warped_img1 = backward_warp(img1, flow[:, 2:4])

                if self.motion:
                    t= timestep.to(device=af[-1-i][:B].device).type(af[-1-i][:B].dtype)

                    flow_, mask_ = self.block[i](
                        torch.cat([t*mf[-1-i][:B],(1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1),
                        torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                        flow
                        )
                else:
                    t=torch.ones_like(imgs[:,:1],device=imgs.device)*timestep.to(device=imgs.device).type(imgs.dtype)

                    flow_, mask_ = self.block[i](
                        torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                        torch.cat((img0, img1, warped_img0, warped_img1, mask,t), 1),
                        flow
                        )

                flow = flow + flow_
                mask = mask + mask_
            else:

                if self.motion:
                    t= timestep.to(device=af[-1-i][:B].device).type(af[-1-i][:B].dtype)

                    flow, mask = self.block[i](
                        torch.cat([t*mf[-1-i][:B],(1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1),
                        torch.cat((img0, img1), 1),
                        None
                        )
                else:
                    t=torch.ones_like(imgs[:,:1],device=imgs.device)*timestep.to(device=imgs.device).type(imgs.dtype)

                    flow, mask = self.block[i](
                        torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                        torch.cat((img0, img1,t), 1),
                        None
                        )

        return flow, mask
    
    def calculate_flow_hr(self, imgs, timestep, af=None, mf=None):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None
        # appearence_features & motion_features
        if self.motion:
            if (af is None) or (mf is None):
                af, mf = self.feature_bone(img0, img1)
        else :
            if (af is None) :
                af = self.feature_bone(img0, img1)
        for i in range(self.flow_num_stage):
            #t = torch.full(mf[-1-i][:B].shape, timestep, dtype=torch.float).cuda()

            if flow != None:
                warped_img0 = backward_warp(img0, flow[:, :2])
                warped_img1 = backward_warp(img1, flow[:, 2:4])

                if self.motion :
                    t= timestep.to(device=af[-1-i][:B].device).type(af[-1-i][:B].dtype)

                    flow_, mask_ = self.block[i](
                        torch.cat([t*mf[-1-i][:B],(1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1),
                        torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                        flow
                        )
                else:
                    t=torch.ones_like(imgs[:,:1],device=imgs.device)*timestep.to(device=imgs.device).type(imgs.dtype)

                    flow_, mask_ = self.block[i](
                        torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                        torch.cat((img0, img1, warped_img0, warped_img1, mask,t), 1),
                        flow
                        )

                flow = flow + flow_
                mask = mask + mask_
            else:
                if self.motion:
                    t= timestep.to(device=af[-1-i][:B].device).type(af[-1-i][:B].dtype)
                    flow, mask = self.block[i](
                        torch.cat([t*mf[-1-i][:B],(1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1),
                        torch.cat((img0, img1), 1),
                        None
                        )
                else:
                    t=torch.ones_like(imgs[:,:1],device=imgs.device)*timestep.to(device=imgs.device).type(imgs.dtype)
                    flow, mask = self.block[i](
                        torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                        torch.cat((img0, img1, t), 1),
                        None
                        )
                
        return flow, mask, af
    
    def calculate_multi_t(self, imgs, step=32, af=None, mf=None):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None

        flow_list_t0 ,flow_list_t1 ,mask_list=[],[],[]
        # appearence_features & motion_features
        if self.motion:
            if (af is None) or (mf is None):
                af, mf = self.feature_bone(img0, img1)
        else :
            if (af is None) :
                af = self.feature_bone(img0, img1)

        for timestep in torch.linspace(1e-6, 1.0, steps=step, device='cuda') :
            timestep=torch.tensor(timestep).reshape(1, 1, 1).unsqueeze(0).cuda()
            
            print(timestep)
            flow=None
            for i in range(self.flow_num_stage):
                #t = torch.full(mf[-1-i][:B].shape, timestep, dtype=torch.float).cuda()

                if flow != None:
                    warped_img0 = backward_warp(img0, flow[:, :2])
                    warped_img1 = backward_warp(img1, flow[:, 2:4])

                    if self.motion :
                        t= timestep.to(device=af[-1-i][:B].device).type(af[-1-i][:B].dtype)

                        flow_, mask_ = self.block[i](
                            torch.cat([t*mf[-1-i][:B],(1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1),
                            torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                            flow
                            )
                    else:
                        t=torch.ones_like(imgs[:,:1],device=imgs.device)*timestep.to(device=imgs.device).type(imgs.dtype)

                        flow_, mask_ = self.block[i](
                            torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                            torch.cat((img0, img1, warped_img0, warped_img1, mask,t), 1),
                            flow
                            )

                    flow = flow + flow_
                    mask = mask + mask_
                else:
                    if self.motion:
                        t= timestep.to(device=af[-1-i][:B].device).type(af[-1-i][:B].dtype)
                        flow, mask = self.block[i](
                            torch.cat([t*mf[-1-i][:B],(1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1),
                            torch.cat((img0, img1), 1),
                            None
                            )
                    else:
                        t=torch.ones_like(imgs[:,:1],device=imgs.device)*timestep.to(device=imgs.device).type(imgs.dtype)
                        flow, mask = self.block[i](
                            torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                            torch.cat((img0, img1, t), 1),
                            None
                            )
            
            flow_list_t0.append(flow[:,:2].unsqueeze(2))
            flow_list_t1.append(flow[:,2:4].unsqueeze(2))
            mask_list.append(mask.unsqueeze(2))

        flowt0  = torch.cat(flow_list_t0,dim=2) # B,2,t,H,W
        flowt1  = torch.cat(flow_list_t1,dim=2)# B,2,t,H,W
        mask    = torch.cat(mask_list,dim=2)# B,1,t,H,W

        if self.motion :
            return  flowt0, flowt1,mask, af, mf
        else:
            return  flowt0, flowt1,mask, af , None

    def coraseWarp_and_Refine(self, imgs, af, flow, mask):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        warped_img0 = backward_warp(img0, flow[:, :2])
        warped_img1 = backward_warp(img1, flow[:, 2:4])

        if self.refine :

            dimgs,dflows,dmasks= self.u_pyramid_features(imgs,flow,mask,level=[1.0,0.5,0.25])
            dwimgs = self.warp_imgs(dimgs, dflows)
            Cs = self.warp_features(af[:-2], dflows)
            res=(self.unet(dimgs, dwimgs, dmasks, dflows, Cs)*2)-1
     
            mask_ = torch.sigmoid(mask)
            merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
            pred = torch.clamp(merged + res,0, 1)

        else:

            mask_ = torch.sigmoid(mask)
            merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
            pred = torch.clamp(merged , 0, 1)

        return pred


    # Actually consist of 'calculate_flow' and 'coraseWarp_and_Refine'
    def forward(self, imgs, gt, timestep=0.5):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = imgs.size(0)
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
    
        af = self.feature_bone(img0, img1)
       
        for i in range(self.flow_num_stage):
            if flow != None:

                t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
                flow_d, mask_d = self.block[i]( torch.cat([af[-1-i][:B],af[-1-i][B:]],1), 
                                                    torch.cat((img0, img1, warped_img0, warped_img1, mask,t), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d
    
            else:
                t=torch.ones_like(img0[:,:1],device=img0.device)*timestep.to(device=img0.device)
                flow, mask = self.block[i]( torch.cat([af[-1-i][:B],af[-1-i][B:]],1), 
                                                torch.cat((img0, img1,t), 1), None)

            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = backward_warp(img0, flow[:, :2])
            warped_img1 = backward_warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_list[-1] + warped_img1 * (1-mask_list[-1]))
            
        if self.refine :
 
            dimgs,dflows,dmasks= self.u_pyramid_features(imgs,flow,mask,level=[1.0,0.5,0.25])
            dwimgs = self.warp_imgs(dimgs, dflows)
            Cs = self.warp_features(af[:-self.flow_num_stage], dflows)
            res=(self.unet(dimgs, dwimgs, dmasks, dflows, Cs)*2)-1
            pred = torch.clamp(merged[-1]+res , 0, 1)
        else:
            pred = torch.clamp(merged[-1] , 0, 1)
        
        return flow_list, mask_list, merged, pred
