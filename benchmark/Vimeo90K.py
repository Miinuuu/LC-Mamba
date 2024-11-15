import cv2
import math
import sys
import torch
import numpy as np
import argparse
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.pytorch_msssim import ssim_matlab

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='my', type=str)
parser.add_argument('--resume', default='my_18_32.90', type=str, help='resume')
parser.add_argument('--path', default='/data/datasets/vimeo_dataset/vimeo_triplet',type=str, required=False)
args = parser.parse_args()


'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
elif args.model == 'ours' :
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )

elif args.model == 'my' :
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'my'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )

model = Model(-1)

epoch,global_step=model.load_checkpoint(args.resume,-1,False)
model.eval()
model.device()

h=None
#h,w=1024,2048

print(f'=========================Starting testing=========================')
print(f'Dataset: Vimeo90K   Model: {model.name}   TTA: {TTA}')
path = args.path
f = open(path + '/tri_testlist.txt', 'r')
psnr_list, ssim_list = [], []
for i in tqdm(f):
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    I0 = cv2.imread(path + '/sequences/' + name + '/im1.png')
    I1 = cv2.imread(path + '/sequences/' + name + '/im2.png')
    I2 = cv2.imread(path + '/sequences/' + name + '/im3.png') # BGR -> RBGW

    if h :
        I0=cv2.resize(I0,(w,h) )
        I1=cv2.resize(I1,(w,h) )
        I2=cv2.resize(I2,(w,h) )

    I0 = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    #print(I0)
    timestep = torch.tensor(0.5).reshape(1, 1, 1)
    mid = model.inference(I0, I2, timestep=timestep, TTA=TTA, fast_TTA=TTA)[0]
    ssim = ssim_matlab(torch.tensor(I1.transpose(2, 0, 1)).cuda().unsqueeze(0) / 255., mid.unsqueeze(0)).detach().cpu().numpy()
    mid = mid.detach().cpu().numpy().transpose(1, 2, 0) 
    I1 = I1 / 255.
    psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)


print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
