import cv2
import torch
import torch.distributed
import numpy as np
import os
import skimage
import glob
from tqdm import tqdm
from benchmark.utils.pytorch_msssim import ssim_matlab
from benchmark.utils.padder import InputPadder
from benchmark.utils.yuv_frame_io import YUV_Read
from skimage.color import rgb2yuv
import logging
import math
import argparse
from Trainer import *
from config import *
from model.feature_extractor import * 
from model.flow_estimation import *
import wandb as wandb
import hashlib
from pathlib import Path
from torchvision.utils import make_grid
from model.visualize import flow2rgb_tensor
from model.visualize import flow2rgb_tensor,norm

import lpips
from flolpips.flolpips import Flolpips



def log_writer(dataset,pred_list, gt_list, overlay_list, flow_list,mask_list,psnr_list,ssim_list,lpips_list,flolpips_list,padder=None):
            
            pred_list=torch.concat(pred_list,0)
            gt_list=torch.concat(gt_list,0)
            overlay_list=torch.concat(overlay_list,0)
            pred_diff= torch.abs(pred_list-gt_list)
            flow_list=torch.concat(flow_list,0)
            masks=torch.concat(mask_list,0) # B,2,H,W

            flowt0 = flow2rgb_tensor(flow_list[:,0:2])
            flowt1 = flow2rgb_tensor(flow_list[:,2:4])
            
            if padder != None:
                flowt0 = padder.unpad(flowt0)
                flowt1 = padder.unpad(flowt1)
                masks = padder.unpad(masks)

            _,c,h,w=overlay_list.shape

            preds = torch.cat([
                        overlay_list.unsqueeze(1),
                        gt_list.unsqueeze(1),
                        pred_list.unsqueeze(1),
                        pred_diff.unsqueeze(1),
                        flowt0.unsqueeze(1),
                        flowt1.unsqueeze(1),
                        masks.clip(0,1).repeat(1,3,1,1).unsqueeze(1),
                        
                        ],dim=1)

            
            preds=(make_grid(preds.reshape(-1,c,h,w)[:,[2,1,0]], nrow=preds.shape[1],value_range=(0,1)))    
        
            run.log({
                'epoch' : 1,
                dataset+'_Avg PSNR': np.mean(psnr_list) , 
                dataset+'_Avg SSIM': np.mean(ssim_list) , 
                dataset+'_Avg LPIPS': np.mean(lpips_list) , 
                dataset+'_Avg FloLPIPS': np.mean(flolpips_list) , 
                dataset+'_preds': wandb.Image(preds, file_type="jpg",mode='RGB',caption=dataset+'_preds'),
                })
            
            del overlay_list 
            del gt_list 
            del pred_list 
            del flowt0 
            del flowt1 
            del preds
            del pred_diff

            torch.cuda.empty_cache()


def bench (model,args ):
        
    path= './benchlog/'
    if not os.path.exists(path):
            os.makedirs(path)
    
    logging.basicConfig(
        filename=path + args.model+'_benchmark.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(args.model+'_experiment.log')
    logger.setLevel(logging.INFO)  # 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.info({ "model" : args.model , "ckpt" : args.resume })

    if  'Xiph' in args.bench:
        print(f'=========================Starting testing=========================')
        print(f'Dataset: Xiph   Model: {args.model} TTA: {False}')
        #path = '/data/dataset/Xiph/test_4k'
        path = os.path.join(args.datasets_path,'Xiph/test_4k')
   
        w_img_n=1
        w_img_n_start=0
        w_img_n_end=w_img_n_start+w_img_n

        down_scale=0.5
        for strCategory in ['resized','cropped']:
            fltPsnr, fltSsim ,fltLpips,fltfloLpips= [], [],[],[]
            npyEstimate_list=[]
            npyReference_list=[]
            overlay_list=[]
            flow_list=[]
            mask_list=[]

            for strFile in tqdm(['BoxingPractice', 'Crosswalk', 'DrivingPOV', 'FoodMarket', 'FoodMarket2','RitualDance', 'SquareAndTimelapse', 'Tango']): 
                for i,intFrame in enumerate(range(2, 99, 2)):
                    npyFirst = cv2.imread(filename=path + '/' + strFile + '/' + str(intFrame - 1).zfill(3) + '.png', flags=-1)
                    npyReference = cv2.imread(filename=path + '/' + strFile + '/' + str(intFrame).zfill(3) + '.png', flags=-1)
                    npySecond = cv2.imread(filename=path + '/' + strFile + '/' + str(intFrame + 1).zfill(3) + '.png', flags=-1)
                    if strCategory == 'resized':
                        npyFirst = cv2.resize(src=npyFirst, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                        npySecond = cv2.resize(src=npySecond, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                        npyReference = cv2.resize(src=npyReference, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

                    elif strCategory == 'cropped':
                        #npyFirst = cv2.resize(src=npyFirst, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                        #npySecond = cv2.resize(src=npySecond, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                        #npyReference = cv2.resize(src=npyReference, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                        npyFirst = npyFirst[540:-540, 1024:-1024, :]
                        npySecond = npySecond[540:-540, 1024:-1024, :]
                        npyReference = npyReference[540:-540, 1024:-1024, :]

                    timestep = torch.tensor(0.5).reshape(1, 1, 1).unsqueeze(0).cuda()
                    tenFirst = torch.FloatTensor(np.ascontiguousarray(npyFirst.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).unsqueeze(0).cuda()
                    tenSecond = torch.FloatTensor(np.ascontiguousarray(npySecond.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).unsqueeze(0).cuda()
                    tenGt = torch.FloatTensor(np.ascontiguousarray(npyReference.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).unsqueeze(0).cuda()
                    
                    if i>= w_img_n_start and i < w_img_n_end :
                        overlay_list.append(timestep*tenFirst+(1-timestep)*tenSecond)
                        npyReference_list.append(tenGt)

                    padder = InputPadder(tenFirst.shape)
                    tenFirst_p, tenSecond_p ,tenGt_p= padder.pad(tenFirst, tenSecond,tenGt)
                    npyEstimate,flow,mask =model.hr_inference(tenFirst_p,tenGt_p, tenSecond_p, timestep=timestep,down_scale=down_scale)
                    npyEstimate=npyEstimate.clamp(0.0, 1.0)
                    npyEstimate = padder.unpad(npyEstimate)
                    lpips= lpips_fn(tenGt*2-1,npyEstimate*2-1).item()
                    flolpips= flolpips_fn(tenFirst,tenSecond,npyEstimate,tenGt).item()
                    #flolpips= flolpips_fn(tenFirst*2-1,tenSecond*2-1,npyEstimate*2-1,tenGt*2-1).item()
                    
                    if i>= w_img_n_start and i < w_img_n_end :
                        npyEstimate_list.append(npyEstimate)
                        flow_list.append(flow)
                        mask_list.append(mask)

                    npyEstimate = (npyEstimate[0].cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

                    psnr = skimage.metrics.peak_signal_noise_ratio(image_true=npyReference, image_test=npyEstimate, data_range=255)
                    ssim = skimage.metrics.structural_similarity(im1=npyReference, im2=npyEstimate, data_range=255,channel_axis=2, multichannel=True)
                    #ssim=ssim_matlab(npyEstimate,npyReference).detach().cpu().numpy()
                    fltPsnr.append(psnr)
                    fltSsim.append(ssim)
                    fltLpips.append(lpips)
                    fltfloLpips.append(flolpips)
                    print('\r {} frame:{} psnr:{} ssim:{} lpips:{} flolpips:{}'.format(strFile, intFrame, psnr, ssim, lpips,flolpips), end = '')

            npyEstimate_list=torch.concat(npyEstimate_list,0)
            npyReference_list=torch.concat(npyReference_list,0)
            overlay_list=torch.concat(overlay_list,0)
            pred_diff= torch.abs(npyEstimate_list-npyReference_list)
            flow_list=torch.concat(flow_list,0)
            masks=torch.concat(mask_list,0) # B,2,H,W

            flowt0 = flow2rgb_tensor(flow_list[:,0:2])
            flowt0 = padder.unpad(flowt0)
            flowt1 = flow2rgb_tensor(flow_list[:,2:4])
            flowt1 = padder.unpad(flowt1)
            masks = padder.unpad(masks)

            _,c,h,w=overlay_list.shape

            preds = torch.cat([
                        overlay_list.unsqueeze(1),
                        npyReference_list.unsqueeze(1),
                        npyEstimate_list.unsqueeze(1),
                        pred_diff.unsqueeze(1),
                        flowt0.unsqueeze(1),
                        flowt1.unsqueeze(1),
                        masks.clip(0,1).repeat(1,3,1,1).unsqueeze(1),
                        
                        ],dim=1)

            preds=(make_grid(preds.reshape(-1,c,h,w)[:,[2,1,0]], nrow=preds.shape[1],value_range=(0,1)))    

            run.log({
                'epoch' : 1,
                'Xiph_'+strCategory+'_Avg PSNR': np.mean(fltPsnr) , 
                'Xiph_'+strCategory+'_Avg SSIM': np.mean(fltSsim) , 
                'Xiph_'+strCategory+'_Avg LPIPS': np.mean(fltLpips) , 
                'Xiph_'+strCategory+'_Avg FloLPIPS': np.mean(fltfloLpips) , 
                'Xiph_'+strCategory+'_preds': wandb.Image(preds, file_type="jpg",mode='RGB',caption=strCategory+'_preds'),
                })
                
            if strCategory == 'resized':
                print('\n---2K---')
                logger.info({'Dataset': 'Xiph-2K', 'Avg PSNR' : np.mean(fltPsnr) ,  "Avg SSIM" : np.mean(fltSsim)   ,"Avg LPIPS" : np.mean(fltLpips) ,"Avg FloLPIPS" : np.mean(fltfloLpips)} )
            
            else:
                print('\n---4K---')
                logger.info({'Dataset': 'Xiph-4K', 'Avg PSNR' : np.mean(fltPsnr) ,  "Avg SSIM" : np.mean(fltSsim) ,"Avg LPIPS" : np.mean(fltLpips),"Avg FloLPIPS" : np.mean(fltfloLpips)} )
            
            del overlay_list 
            del npyReference_list 
            del npyEstimate_list 
            del flowt0 
            del flowt1 
            del preds
            del pred_diff
            torch.cuda.empty_cache()
            print('Avg PSNR:', np.mean(fltPsnr))
            print('Avg SSIM:', np.mean(fltSsim))
            print('Avg LPIPS:', np.mean(fltLpips))
            print('Avg FloLPIPS:', np.mean(fltfloLpips))

    if  'XTest_8X' in args.bench:
        print(f'=========================Starting testing=========================')
        print(f'Dataset: XTest_8X   Model: {args.model}   TTA: {False}')
        def getXVFI(dir, multiple=8, t_step_size=32):
            """ make [I0,I1,It,t,scene_folder] """
            testPath = []
            t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
            for type_folder in sorted(glob.glob(os.path.join(dir, '*', ''))):
                for scene_folder in sorted(glob.glob(os.path.join(type_folder, '*', ''))):
                    frame_folder = sorted(glob.glob(scene_folder + '*.png'))
                    for idx in range(0, len(frame_folder), t_step_size):
                        if idx == len(frame_folder) - 1:
                            break
                        for mul in range(multiple - 1):
                            I0I1It_paths = []
                            I0I1It_paths.append(frame_folder[idx])
                            I0I1It_paths.append(frame_folder[idx + t_step_size])
                            I0I1It_paths.append(frame_folder[idx + int((t_step_size // multiple) * (mul + 1))])
                            I0I1It_paths.append(t[mul])
                            testPath.append(I0I1It_paths)

            return testPath

        data_path = os.path.join(args.datasets_path,'X4K1000FPS/test')
        listFiles = getXVFI(data_path)
        
        
        w_img_n=20
        w_img_n_start=0
        w_img_n_end=w_img_n_start+w_img_n


        for strMode in ['XTEST-2k', 'XTEST-4k']:
            fltPsnr, fltSsim , fltLpips,fltfloLpips= [], [],[],[]
            pred_list=[]
            gt_list=[]
            overlay_list=[]
            flow_list=[]
            mask_list=[]


            for i,intFrame in enumerate(tqdm(listFiles)):
                npyOne = np.array(cv2.imread(intFrame[0])).astype(np.float32) * (1.0 / 255.0)
                npyTwo = np.array(cv2.imread(intFrame[1])).astype(np.float32) * (1.0 / 255.0)
                npyTruth = np.array(cv2.imread(intFrame[2])).astype(np.float32) * (1.0 / 255.0)

                if strMode == 'XTEST-2k': #downsample
                    down_scale = 0.5
                    npyOne = cv2.resize(src=npyOne, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                    npyTwo = cv2.resize(src=npyTwo, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                    npyTruth = cv2.resize(src=npyTruth, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                else:
                    down_scale = 0.25

                tenOne = torch.FloatTensor(np.ascontiguousarray(npyOne.transpose(2, 0, 1)[None, :, :, :])).cuda()
                tenTwo = torch.FloatTensor(np.ascontiguousarray(npyTwo.transpose(2, 0, 1)[None, :, :, :])).cuda()
                tenGT = torch.FloatTensor(np.ascontiguousarray(npyTruth.transpose(2, 0, 1)[None, :, :, :])).cuda()
                timestep = torch.tensor(intFrame[3]).reshape(1, 1, 1).unsqueeze(0).cuda()
                
                if i>= w_img_n_start and i < w_img_n_end :
                    if intFrame[3] == 0.5 :
                        overlay_list.append(timestep*tenOne + (1-timestep)*tenTwo)
                        gt_list.append(tenGT)

                padder = InputPadder(tenOne.shape, 32)
                tenOne_p, tenTwo_p,tenGT_p = padder.pad(tenOne, tenTwo,tenGT)
                #timestep= torch.from_numpy( np.expand_dims(np.array(intFrame[3], dtype=np.float32), 0))

                tenEstimate ,flow ,mask= model.hr_inference(tenOne_p,tenGT_p,tenTwo_p, timestep=timestep, down_scale = down_scale)
                tenEstimate = padder.unpad(tenEstimate)

                if i>= w_img_n_start and i < w_img_n_end :
                    if intFrame[3] == 0.5 :
                        pred_list.append(tenEstimate)
                        flow_list.append(flow)
                        mask_list.append(mask)

                #npyEstimate = (tenEstimate.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).round().astype(np.uint8)
                #tenEstimate = torch.FloatTensor(npyEstimate.transpose(2, 0, 1)[None, :, :, :]).cuda() / 255.0

                fltPsnr.append(-10 * math.log10(torch.mean((tenEstimate - tenGT) * (tenEstimate - tenGT)).cpu().data))
                fltSsim.append(ssim_matlab(tenEstimate,tenGT).detach().cpu().numpy())
                fltLpips.append(lpips_fn(tenGT*2-1,tenEstimate*2-1).item())
                fltfloLpips.append(flolpips_fn(tenOne,tenTwo,tenEstimate,tenGT).item())

            log_writer(dataset=strMode,pred_list=pred_list, gt_list=gt_list, overlay_list=overlay_list, flow_list=flow_list ,mask_list=mask_list,psnr_list=fltPsnr,ssim_list=fltSsim, lpips_list= fltLpips,flolpips_list=fltfloLpips, padder=padder)

            print(f'{strMode}  PSNR: {np.mean(fltPsnr)}  SSIM: {np.mean(fltSsim)}, LPIPS: {np.mean(fltLpips)}, Flo_LPIPS: {np.mean(fltfloLpips)}')
            logger.info({'Dataset': str(strMode), 'Avg PSNR' : np.mean(fltPsnr) ,  "Avg SSIM" : np.mean(fltSsim),  "Avg LPIPS" : np.mean(fltLpips),  "Avg FloLPIPS" : np.mean(fltfloLpips)} )

    if  'HD_4X' in args.bench:
        print(f'=========================Starting testing=========================')
        print(f'Dataset: HD_4X   Model: {args.model}   TTA: {False}')
        #path = '/data/dataset/HD_dataset'
        path=os.path.join(args.datasets_path,'HD_dataset')
        down_scale=1.0
        name_list = [
            ('HD720p_GT/parkrun_1280x720_50.yuv', 720, 1280),
            ('HD720p_GT/shields_1280x720_60.yuv', 720, 1280),
            ('HD720p_GT/stockholm_1280x720_60.yuv', 720, 1280),
            ('HD1080p_GT/BlueSky.yuv', 1080, 1920),
            ('HD1080p_GT/Kimono1_1920x1080_24.yuv', 1080, 1920),
            ('HD1080p_GT/ParkScene_1920x1080_24.yuv', 1080, 1920),
            ('HD1080p_GT/sunflower_1080p25.yuv', 1080, 1920),
            ('HD544p_GT/Sintel_Alley2_1280x544.yuv', 544, 1280),
            ('HD544p_GT/Sintel_Market5_1280x544.yuv', 544, 1280),
            ('HD544p_GT/Sintel_Temple1_1280x544.yuv', 544, 1280),
            ('HD544p_GT/Sintel_Temple2_1280x544.yuv', 544, 1280),
        ]

                
        tot = []
        for data in tqdm(name_list):
            psnr_list = []
            name = data[0]
            h, w = data[1], data[2]
            Reader = YUV_Read(os.path.join(path, name), h, w, toRGB=True)
            _, lastframe = Reader.read()

            for index in tqdm(range(0, 100, 4)):
                gt = []
                IMAGE1, success1 = Reader.read(index)
                IMAGE2, success2 = Reader.read(index + 4)
                if not success2:
                    break
                for i in range(1, 4):
                    tmp, _ = Reader.read(index + i)
                    gt.append(tmp)

                I0 = torch.from_numpy(np.transpose(IMAGE1, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
                I1 = torch.from_numpy(np.transpose(IMAGE2, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
                
                padder = InputPadder(I0.shape, divisor=32)
                I0_p, I1_p = padder.pad(I0, I1)
            
            
               #[   (i+1)*(1./4.) for i in range(3)     ]
                timestep= [torch.tensor((i+1)*(1./4.)).reshape(1, 1, 1) for i in range(3)]
 
                #timestep=torch.from_numpy( np.expand_dims(np.array( (i+1)*(1./4.), dtype=np.float32), 0))
                pred_list ,flow_list,maks_list= model.multi_inference(I0_p, I1_p, TTA=TTA, time_list=timestep, fast_TTA = TTA)
            
                for i in range(len(pred_list)):
                    pred_list[i] = padder.unpad(pred_list[i])
                    flow_list[i] = padder.unpad(flow_list[i])
                    maks_list[i] = padder.unpad(maks_list[i])

                for i in range(3):
                    out = (np.round(pred_list[i].detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')
                    diff_rgb = 128.0 + rgb2yuv(gt[i] / 255.)[:, :, 0] * 255 - rgb2yuv(out / 255.)[:, :, 0] * 255
                    mse = np.mean((diff_rgb - 128.0) ** 2)
                    PIXEL_MAX = 255.0
                    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

                    psnr_list.append(psnr)

            tot.append(np.mean(psnr_list))

        print('PSNR: {}(544*1280), {}(720p), {}(1080p)'.format(np.mean(tot[7:11]), np.mean(tot[:3]), np.mean(tot[3:7])))
        logger.info({'Dataset': 'HD_4X' , '(544*1280) Avg PSNR' : np.mean(tot[7:11]) ,  '(720p) Avg PSNR' : np.mean(tot[:3]) ,  '(1080p) Avg PSNR' : np.mean(tot[3:7])} )
        
    if  'Vimeo90K' in args.bench:

        print(f'=========================Starting testing=========================')
        print(f'Dataset: Vimeo90K triplet   Model: {args.model}   TTA: {False}')
        #path = '/data/dataset/vimeo_dataset/vimeo_triplet'
        path=os.path.join(args.datasets_path,'vimeo_triplet')
        f = open(path + '/tri_testlist.txt', 'r')
        psnr_list, ssim_list ,lpips_list,flolpips_list= [], [],[],[]
        pred_list,flow_list=[],[]
        overlay_list=[]
        mask_list=[]

        gt_list=[]
        w_img_n=100
        w_img_n_start=0
        w_img_n_end=w_img_n_start+w_img_n
        for n,i in enumerate(tqdm(f)):
            name = str(i).strip()
            if(len(name) <= 1):
                continue
            I0 = cv2.imread(path + '/sequences/' + name + '/im1.png')
            I1 = cv2.imread(path + '/sequences/' + name + '/im2.png')
            I2 = cv2.imread(path + '/sequences/' + name + '/im3.png') # BGR -> RBGW
            I0 = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
            gt = (torch.tensor(I1.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
            I2 = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
            timestep = torch.tensor(0.5).reshape(1, 1, 1).unsqueeze(0).cuda()
            mid ,flow,mask= model.inference(I0,gt,I2,timestep=timestep)

            if n >= w_img_n_start and n < w_img_n_end :
                pred_list.append(mid)
                flow_list.append(flow)
                mask_list.append(mask)
                gt_list.append(gt)
                overlay_list.append(timestep*I0+(1-timestep)*I2)

            lpips_list.append(lpips_fn(mid*2-1,gt*2-1).item())
            #flolpips_list.append(flolpips_fn(I0*2-1,I2*2-1,mid*2-1,gt*2-1).item())
            flolpips_list.append(flolpips_fn(I0,I2,mid,gt).item())
            mid=mid[0]
            ssim = ssim_matlab(torch.tensor(I1.transpose(2, 0, 1)).cuda().unsqueeze(0) / 255., mid.unsqueeze(0)).detach().cpu().numpy()
            mid = mid.detach().cpu().numpy().transpose(1, 2, 0) 
            I1 = I1 / 255.
            psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        overlay_list=torch.concat(overlay_list,dim=0)
        gt_list=torch.concat(gt_list,dim=0)
        pred_list=torch.concat(pred_list,dim=0)
        flow_list=torch.concat(flow_list,dim=0)
        pred_diff =torch.abs(pred_list-gt_list)
        flowt0 = flow2rgb_tensor(flow_list[:,0:2])
        flowt1 = flow2rgb_tensor(flow_list[:,2:4])
        _,c,h,w=flowt1.shape
        masks=torch.concat(mask_list,0) # B,2,H,W

        preds = torch.cat([
                    overlay_list.unsqueeze(1),
                    gt_list.unsqueeze(1),
                    pred_list.unsqueeze(1),
                    pred_diff.unsqueeze(1),
                    flowt0.unsqueeze(1),
                    flowt1.unsqueeze(1),
                    masks.clip(0,1).repeat(1,3,1,1).unsqueeze(1),
                           
                    ],dim=1)
                                
        preds=(make_grid(preds.reshape(-1,c,h,w)[:,[2,1,0]], nrow=preds.shape[1],value_range=(0,1)))    
    
        run.log({
            'epoch' : 1,
            'Vimeo90k_Avg PSNR': np.mean(psnr_list) , 
            'Vimeo90k_Avg SSIM': np.mean(ssim_list) , 
            'Vimeo90k_Avg LPIPS': np.mean(lpips_list) , 
            'Vimeo90k_Avg FloLPIPS': np.mean(flolpips_list) , 
            'Vimeo90k_preds': wandb.Image(preds, file_type="jpg",mode='RGB',caption='Vimeo90k_preds'),
            })
        


        print("Avg PSNR: {} SSIM: {} LPIPS {} FloLPIPS {}".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list), np.mean(flolpips_list)))
        logger.info({'Dataset': 'Vimeo90K', 'Avg PSNR' : np.mean(psnr_list),"Avg SSIM" : np.mean(ssim_list), "Avg LPIPS" : np.mean(lpips_list), "Avg FloLPIPS" : np.mean(flolpips_list)})

        del overlay_list 
        del gt_list 
        del pred_list 
        del flowt0 
        del flowt1 
        del preds
        del pred_diff
        torch.cuda.empty_cache()

    if  'UCF101' in args.bench:
        print(f'=========================Starting testing=========================')
        print(f'Dataset: UCF101   Model: {args.model}   TTA: {False}')
            
        #path = '/data/dataset/ucf101'
        path=os.path.join(args.datasets_path,'ucf101')
        print(path)
        dirs = os.listdir(path)
        psnr_list, ssim_list ,lpips_list,floLpips_list = [], [], [], []
        for d in tqdm(dirs):
            img0 = (path + '/' + d + '/frame_00.png')
            img1 = (path + '/' + d + '/frame_02.png')
            gt = (path + '/' + d + '/frame_01_gt.png')
            img0 = (torch.tensor(cv2.imread(img0).transpose(2, 0, 1) / 255.)).cuda().float().unsqueeze(0)
            img1 = (torch.tensor(cv2.imread(img1).transpose(2, 0, 1) / 255.)).cuda().float().unsqueeze(0)
            gt = (torch.tensor(cv2.imread(gt).transpose(2, 0, 1) / 255.)).cuda().float().unsqueeze(0)
            timestep = torch.tensor(0.5).reshape(1, 1, 1).unsqueeze(0).cuda()
            pred,flow,mask = model.inference(img0,gt, img1, timestep=timestep)
            lpips_list.append(lpips_fn(pred*2-1,gt*2-1).item())
            floLpips_list.append(flolpips_fn(img0,img1,pred,gt).item())

            pred=pred[0]
            ssim = ssim_matlab(gt, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
            out = pred.detach().cpu().numpy().transpose(1, 2, 0)
            out = np.round(out * 255) / 255.
            gt = gt[0].cpu().numpy().transpose(1, 2, 0)
            psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())
            psnr_list.append(psnr)
            ssim_list.append(ssim)
        print("Avg PSNR: {} SSIM: {} LPIPS: {} FloLPIPS : {}".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list), np.mean(floLpips_list)))
        logger.info({'Dataset': 'UCF101', 
                     'Avg PSNR' : np.mean(psnr_list) ,  
                     "Avg SSIM" : np.mean(ssim_list),
                     "Avg LPIPS" : np.mean(lpips_list),
                     "Avg FloLPIPS" : np.mean(floLpips_list),
                     } )


    if  'SNU_FILM' in args.bench:
        print(f'=========================Starting testing=========================')
        print(f'Dataset: SNU_FILM   Model: {args.model}   TTA: {False}')
        #path = '/data/dataset/snufilm'
        path=os.path.join(args.datasets_path,'snufilm')
        down_scale = 0.5

        
        w_img_n=4
        w_img_n_start=0
        w_img_n_end=w_img_n_start+w_img_n


        level_list = ['test-easy.txt', 'test-medium.txt', 'test-hard.txt', 'test-extreme.txt'] 
        for test_file in level_list:
            psnr_list, ssim_list ,lpips_list,flolpips_list= [], [], [],[]
            file_list = []
            
            pred_list=[]
            gt_list=[]
            overlay_list=[]
            flow_list=[]
            mask_list=[]
     
            with open(os.path.join(path,'eval_modes',test_file), "r") as f:
                for line in f:
                    line = line.strip()
                    #print(line)
                    file_list.append(line.split(' '))

            for i,line in enumerate(tqdm(file_list)):
                #print(line)
                I0_path = os.path.join(path, line[0])
                I1_path = os.path.join(path, line[1])
                I2_path = os.path.join(path, line[2])
                I0 = cv2.imread(I0_path)
                I1_ = cv2.imread(I1_path)
                I2 = cv2.imread(I2_path)
                I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
                I1 = (torch.tensor(I1_.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
                I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()

                timestep = torch.tensor(0.5).reshape(1, 1, 1).unsqueeze(0).cuda()
                if i>= w_img_n_start and i < w_img_n_end :
                    overlay_list.append(timestep*I0 + (1-timestep)*I2)
                    gt_list.append(I1)



                padder = InputPadder(I0.shape, divisor=32)
                I0_p, I2_p ,I1_p= padder.pad(I0, I2,I1)
                I1_pred ,flow,mask= model.hr_inference(I0_p, I1_p, I2_p, timestep=timestep, down_scale = down_scale)
                I1_pred = padder.unpad(I1_pred)
                lpips_list.append(lpips_fn(I1_pred*2-1,I1*2-1).item())
                flolpips_list.append(flolpips_fn(I0,I2,I1_pred,I1).item())
                if i>= w_img_n_start and i < w_img_n_end :
                    pred_list.append(I1_pred)
                    flow_list.append(flow)
                    mask_list.append(mask)

                ssim = ssim_matlab(I1, I1_pred).detach().cpu().numpy()

                I1_pred = I1_pred[0].detach().cpu().numpy().transpose(1, 2, 0)   
                I1_ = I1_ / 255.
                #print(I1_.shape, I1_pred.shape)
                psnr = -10 * math.log10(((I1_ - I1_pred) * (I1_ - I1_pred)).mean())
                
                psnr_list.append(psnr)
                ssim_list.append(ssim)

            log_writer(dataset='SNU_FILM_'+test_file[:-4],pred_list=pred_list, gt_list=gt_list, overlay_list=overlay_list, flow_list=flow_list ,mask_list=mask_list,psnr_list=psnr_list,ssim_list=ssim_list,lpips_list=lpips_list,flolpips_list=flolpips_list,padder=padder)

            print('Testing level:' + test_file[:-4])
            print('Avg PSNR: {} SSIM: {} LPIPS {} FloLPIPS {}'.format(np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list), np.mean(flolpips_list)))
            logger.info({'Dataset': 'SNU_FILM'+test_file[:-4], 'Avg PSNR' : np.mean(psnr_list) ,  "Avg SSIM" : np.mean(ssim_list),  "Avg LPIPS" : np.mean(lpips_list),  "Avg FloLPIPS" : np.mean(flolpips_list)} )

    if  'MiddleBury' in args.bench:
        print(f'=========================Starting testing=========================')
        print(f'Dataset: MiddleBury   Model: {args.model}   TTA: {False}')
        #path = '/data/dataset/middlebury'
        path= os.path.join(args.datasets_path,'middlebury')
        name = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        IE_list = []
        for i in tqdm(name):
            i0 = cv2.imread(path + '/other-data/{}/frame10.png'.format(i)).transpose(2, 0, 1) / 255.
            i1 = cv2.imread(path + '/other-gt-interp/{}/frame10i11.png'.format(i)) 
            i2 = cv2.imread(path + '/other-data/{}/frame11.png'.format(i)).transpose(2, 0, 1) / 255.
            gt=i1
            i0 = torch.from_numpy(i0).unsqueeze(0).float().cuda()
            i1 = torch.from_numpy(i1).unsqueeze(0).float().cuda()
            i2 = torch.from_numpy(i2).unsqueeze(0).float().cuda()
            
            padder = InputPadder(i0.shape, divisor = 32)
            i0_p, i1_p ,i2_p= padder.pad(i0,i1,i2)
            
            timestep = torch.tensor(0.5).reshape(1, 1, 1).unsqueeze(0).cuda()
            pred,flow,mask= model.inference(i0_p, i1_p, i2_p, timestep=timestep)
            pred=pred[0]
            pred = padder.unpad(pred)
            out = pred.detach().cpu().numpy().transpose(1, 2, 0)
            out = np.round(out * 255.)
            IE_list.append(np.abs((out - gt * 1.0)).mean())
        print(f"Avg IE: {np.mean(IE_list)}")
        logger.info({'Dataset': 'MiddleBury', 'Avg IE' : np.mean(IE_list)} )
        logger.info("---------------------end-----------------------")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    #parser.add_argument('--bench'           ,default=[ 'Xiph', 'XTest_8X','Vimeo90K','UCF101','SNU_FILM','MiddleBury','HD_4X' ], type=str)
    parser.add_argument('--bench'           ,default=['Xiph','Vimeo90K','UCF101','SNU_FILM','MiddleBury'], type=str)
    parser.add_argument('--datasets_path'   ,default='/data/datasets',type=str, help='datasets path')
    parser.add_argument('--model'           ,default='lite'          ,type=str)
    parser.add_argument('--trainer'         ,default='Model'         ,type=str,help='trainer')
    parser.add_argument('--resume'          ,default=None            ,type=str, help='resume')
    parser.add_argument('--strict_model'    ,default=True           ,action='store_true'    ,help='strict model')
    parser.add_argument('--project'         ,default='benchmark'     ,type=str, help='wandb project name')

    args = parser.parse_args()
    '''==========Model setting=========='''
    TTA=False
    MODEL_CONFIG= Model_create(args.model)

    exec(args.trainer, globals())
    created_class = globals()[args.trainer]


    wandb.login()
    log_dir = Path.cwd().absolute() / "wandb_logs" / args.model
    log_dir.mkdir(exist_ok=True, parents=True)
    sha = hashlib.sha256()
    sha.update(str(args.model).encode())
    wandb_id = sha.hexdigest()

    
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval() #alex,vgg,squeeze
    flolpips_fn = Flolpips().cuda().eval()

    if args.resume :
        model = created_class(local_rank=-1,MODEL_CONFIG=MODEL_CONFIG)        
        note = str(model.num_param) +'M'

        run = wandb.init(project=args.project,
                        id=  wandb_id, 
                        dir= log_dir,
                        job_type='bench',
                        save_code=True,
                        notes=note,
                        name=args.model,
                        resume='allow')
        ckpt=model.load_checkpoint(args.resume)
        model.eval()
        model.device()
        
        epoch=ckpt['epoch']
        global_step=ckpt['global_step']
        cur_psnr=['psnr']
        if args.strict_model :
            assert args.model == ckpt['Model']
        bench(model, args)
        run.finish