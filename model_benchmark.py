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
import lpips
from flolpips.flolpips import Flolpips




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

    if  'Vimeo90K' in args.bench:

        print(f'=========================Starting testing=========================')
        print(f'Dataset: Vimeo90K triplet   Model: {args.model}   TTA: {False}')
        #path = '/data/dataset/vimeo_dataset/vimeo_triplet'
        path=os.path.join(args.datasets_path,'vimeo_triplet')
        f = open(path + '/tri_testlist.txt', 'r')
        psnr_list, ssim_list ,lpips_list,flolpips_list= [], [],[],[]

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
            mid  = model.inference(I0, I2,timestep=timestep)


                
            lpips_list.append(lpips_fn(mid*2-1,gt*2-1).item())
            flolpips_list.append(flolpips_fn(I0,I2,mid,gt).item())
            mid=mid[0]
            ssim = ssim_matlab(torch.tensor(I1.transpose(2, 0, 1)).cuda().unsqueeze(0) / 255., mid.unsqueeze(0)).detach().cpu().numpy()
            mid = mid.detach().cpu().numpy().transpose(1, 2, 0) 
            I1 = I1 / 255.
            psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        


        print("Avg PSNR: {} SSIM: {} LPIPS {} FloLPIPS {}".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list), np.mean(flolpips_list)))
        logger.info({'Dataset': 'Vimeo90K', 'Avg PSNR' : np.mean(psnr_list),"Avg SSIM" : np.mean(ssim_list), "Avg LPIPS" : np.mean(lpips_list), "Avg FloLPIPS" : np.mean(flolpips_list)})

        torch.cuda.empty_cache()

    if  'UCF101' in args.bench:
        print(f'=========================Starting testing=========================')
        print(f'Dataset: UCF101   Model: {args.model}   TTA: {False}')
            
        #path = '/data/dataset/ucf101'
        path=os.path.join(args.datasets_path,'ucf101')
        #print(path)
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
            pred = model.inference(img0, img1, timestep=timestep)
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


    if  'Xiph' in args.bench:
        print(f'=========================Starting testing=========================')
        print(f'Dataset: Xiph   Model: {args.model} TTA: {False}')
        path = os.path.join(args.datasets_path,'Xiph/test_4k')
   
        w_img_n=1
        w_img_n_start=0
        w_img_n_end=w_img_n_start+w_img_n

        down_scale=0.5
        for strCategory in ['resized','cropped']:
            fltPsnr, fltSsim ,fltLpips,fltfloLpips= [], [],[],[]
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
                        npyFirst = npyFirst[540:-540, 1024:-1024, :]
                        npySecond = npySecond[540:-540, 1024:-1024, :]
                        npyReference = npyReference[540:-540, 1024:-1024, :]

                    timestep = torch.tensor(0.5).reshape(1, 1, 1).unsqueeze(0).cuda()
                    tenFirst = torch.FloatTensor(np.ascontiguousarray(npyFirst.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).unsqueeze(0).cuda()
                    tenSecond = torch.FloatTensor(np.ascontiguousarray(npySecond.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).unsqueeze(0).cuda()
                    tenGt = torch.FloatTensor(np.ascontiguousarray(npyReference.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).unsqueeze(0).cuda()
        
                    padder = InputPadder(tenFirst.shape)
                    tenFirst_p, tenSecond_p = padder.pad(tenFirst, tenSecond)

                    npyEstimate =model.hr_inference(tenFirst_p,tenSecond_p, timestep=timestep,down_scale=down_scale)
                    npyEstimate=npyEstimate.clamp(0.0, 1.0)
                    npyEstimate = padder.unpad(npyEstimate)
                    lpips= lpips_fn(npyEstimate*2-1,tenGt*2-1).item()
                    flolpips= flolpips_fn(tenFirst,tenSecond,npyEstimate,tenGt).item()
                    
                    npyEstimate = (npyEstimate[0].cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

                    psnr = skimage.metrics.peak_signal_noise_ratio(image_true=npyReference, image_test=npyEstimate, data_range=255)
                    ssim = skimage.metrics.structural_similarity(im1=npyReference, im2=npyEstimate, data_range=255,channel_axis=2, multichannel=True)

                    fltPsnr.append(psnr)
                    fltSsim.append(ssim)
                    fltLpips.append(lpips)
                    fltfloLpips.append(flolpips)
                    print('\r {} frame:{} psnr:{} ssim:{} lpips:{} flolpips:{}'.format(strFile, intFrame, psnr, ssim, lpips,flolpips), end = '')

            
            if strCategory == 'resized':
                print('\n---2K---')
                logger.info({'Dataset': 'Xiph-2K', 'Avg PSNR' : np.mean(fltPsnr) ,  "Avg SSIM" : np.mean(fltSsim)   ,"Avg LPIPS" : np.mean(fltLpips) ,"Avg FloLPIPS" : np.mean(fltfloLpips)} )
            
            else:
                print('\n---4K---')
                logger.info({'Dataset': 'Xiph-4K', 'Avg PSNR' : np.mean(fltPsnr) ,  "Avg SSIM" : np.mean(fltSsim) ,"Avg LPIPS" : np.mean(fltLpips),"Avg FloLPIPS" : np.mean(fltfloLpips)} )
        
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
        
    
        for strMode in ['XTEST-2k', 'XTEST-4k']:
            fltPsnr, fltSsim , fltLpips,fltfloLpips= [], [],[],[]
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
                
        
                padder = InputPadder(tenOne.shape, 32)
                tenOne_p, tenTwo_p = padder.pad(tenOne, tenTwo)

                tenEstimate = model.hr_inference(tenOne_p, tenTwo_p, timestep=timestep, down_scale = down_scale)
                tenEstimate = padder.unpad(tenEstimate)

                fltPsnr.append(-10 * math.log10(torch.mean((tenEstimate - tenGT) * (tenEstimate - tenGT)).cpu().data))
                fltSsim.append(ssim_matlab(tenEstimate,tenGT).detach().cpu().numpy())
                fltLpips.append(lpips_fn(tenEstimate*2-1,tenGT*2-1).item())
                fltfloLpips.append(flolpips_fn(tenOne,tenTwo,tenEstimate,tenGT).item())


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
                        
                timestep= [torch.tensor((i+1)*(1./4.)).reshape(1, 1, 1) for i in range(3)] 
                pred_list = model.multi_inference(I0_p, I1_p, TTA=TTA, time_list=timestep, fast_TTA = TTA)
            
                for i in range(len(pred_list)):
                    pred_list[i] = padder.unpad(pred_list[i])
    

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
        
    if  'SNU_FILM' in args.bench:
        print(f'=========================Starting testing=========================')
        print(f'Dataset: SNU_FILM   Model: {args.model}   TTA: {False}')
        #path = '/data/dataset/snufilm'
        path=os.path.join(args.datasets_path,'snufilm')
        down_scale = 0.5

    

        level_list = ['test-easy.txt', 'test-medium.txt', 'test-hard.txt', 'test-extreme.txt'] 
        for test_file in level_list:
            psnr_list, ssim_list ,lpips_list,flolpips_list= [], [], [],[]
            file_list = []
            
            
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
  

                padder = InputPadder(I0.shape, divisor=32)
                I0_p, I2_p = padder.pad(I0, I2)
                I1_pred = model.hr_inference(I0_p, I2_p, timestep=timestep, down_scale = down_scale)
                I1_pred = padder.unpad(I1_pred)
                lpips_list.append(lpips_fn(I1_pred*2-1,I1*2-1).item())
                flolpips_list.append(flolpips_fn(I0,I2,I1_pred,I1).item())
                
                ssim = ssim_matlab(I1, I1_pred).detach().cpu().numpy()

                I1_pred = I1_pred[0].detach().cpu().numpy().transpose(1, 2, 0)   
                I1_ = I1_ / 255.
                psnr = -10 * math.log10(((I1_ - I1_pred) * (I1_ - I1_pred)).mean())
                
                psnr_list.append(psnr)
                ssim_list.append(ssim)

            print('Testing level:' + test_file[:-4])
            print('Avg PSNR: {} SSIM: {} LPIPS {} FloLPIPS {}'.format(np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list), np.mean(flolpips_list)))
            logger.info({'Dataset': 'SNU_FILM'+test_file[:-4], 'Avg PSNR' : np.mean(psnr_list) ,  "Avg SSIM" : np.mean(ssim_list),  "Avg LPIPS" : np.mean(lpips_list),  "Avg FloLPIPS" : np.mean(flolpips_list)} )

    if  'MiddleBury' in args.bench:
        print(f'=========================Starting testing=========================')
        print(f'Dataset: MiddleBury   Model: {args.model}   TTA: {False}')
        path= os.path.join(args.datasets_path,'middlebury')
        name = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        IE_list = []
        for i in tqdm(name):
            i0 = cv2.imread(path + '/other-data/{}/frame10.png'.format(i)).transpose(2, 0, 1) / 255.
            i1 = cv2.imread(path + '/other-data/{}/frame11.png'.format(i)).transpose(2, 0, 1) / 255.
            gt = cv2.imread(path + '/other-gt-interp/{}/frame10i11.png'.format(i)) 
            i0 = torch.from_numpy(i0).unsqueeze(0).float().cuda()
            i1 = torch.from_numpy(i1).unsqueeze(0).float().cuda()
            padder = InputPadder(i0.shape, divisor = 32)
            i0_p, i1_p = padder.pad(i0, i1)
            
            timestep = torch.tensor(0.5).reshape(1, 1, 1).unsqueeze(0).cuda()
            pred = model.inference(i0_p, i1_p, timestep=timestep)
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
    parser.add_argument('--bench'           ,default=[ 'Xiph','Vimeo90K','UCF101','SNU_FILM','MiddleBury' ], type=str)
    parser.add_argument('--datasets_path'   ,default='/data/datasets'   ,type=str, help='datasets path')
    parser.add_argument('--model'           ,default='Ours-BS'           ,type=str)
    parser.add_argument('--trainer'         ,default='Model'            ,type=str,help='trainer')
    parser.add_argument('--resume'          ,default=None               ,type=str, help='resume')
    parser.add_argument('--strict_model'    ,default=False              ,action='store_true',help='strict model')

    args = parser.parse_args()
    '''==========Model setting=========='''
    TTA=False
    MODEL_CONFIG= Model_create(args.model)

    exec(args.trainer, globals())
    created_class = globals()[args.trainer]
    
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval() #alex,vgg,squeeze
    flolpips_fn = Flolpips().cuda().eval()

    if args.resume :
        model = created_class(local_rank=-1,MODEL_CONFIG=MODEL_CONFIG)        

        note = str(model.num_param) +'M'

        ckpt=model.load_checkpoint(args.resume)
        model.eval()
        model.device()
        
        if args.strict_model :
            assert args.model == ckpt['Model']
        bench(model, args)
