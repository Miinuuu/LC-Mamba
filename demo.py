import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave
from typing import Any, Callable, Optional, Sequence, Union

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import *
from benchmark.utils.padder import InputPadder
from config import *
import hashlib
from pathlib import Path
from torchvision.utils import make_grid


parser = argparse.ArgumentParser()


parser.add_argument('--model', default='Ours-E', type=str)
parser.add_argument('--resume', default=None, type=str, help='resume')
parser.add_argument('--trainer', type=str, default='Model',help='trainer')
parser.add_argument('--size', type=Union[int, Sequence[int]], default=[4096,2048],help='img size')
parser.add_argument('--scale', type=int, default=0.25,help='img scale')
parser.add_argument('--strict_model', action='store_true', default=False,help='strict model')
parser.add_argument('--i0',  default='/home/jmw/ing/EMAA/example/Beanbags/frame07.png',help='image0 path')
parser.add_argument('--i1',  default='/home/jmw/ing/EMAA/example/Beanbags/frame08.png',help='image1 path')
parser.add_argument('--i2',  default='/home/jmw/ing/EMAA/example/Beanbags/frame09.png',help='image2 path')

args = parser.parse_args()

TTA=False
MODEL_CONFIG= Model_create(args.model)
exec(args.trainer, globals())
created_class = globals()[args.trainer]
model = created_class(local_rank=-1,MODEL_CONFIG=MODEL_CONFIG)        
ckpt=model.load_checkpoint(args.resume)
model.eval()
model.device()
epoch=ckpt['epoch']
global_step=ckpt['global_step']
cur_psnr=['psnr']
if args.strict_model :
    assert args.model == ckpt['Model']

w,h = args.size[0],args.size[1]
down_scale=args.scale
print(f'=========================Start Generating=========================')

I0 = cv2.imread(args.i0)
I1 = cv2.imread(args.i1)
I2 = cv2.imread(args.i2)

I0=cv2.cvtColor(I0, cv2.COLOR_BGR2RGB)
I1=cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
I2=cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

I0 = cv2.resize(I0,(w,h))#1k
I1 = cv2.resize(I1,(w,h))#1k
I2 = cv2.resize(I2,(w,h))

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I1_ = (torch.tensor(I1.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_p, I2_p = padder.pad(I0_, I2_)
timestep = torch.tensor(0.5).reshape(1, 1, 1).unsqueeze(0).cuda()


if down_scale != 1.0 :
    print('hr inference')
    pred = (model.hr_inference(I0_p, I2_p, timestep=timestep,TTA=TTA, down_scale=down_scale,fast_TTA=TTA))
    pred = padder.unpad(pred)
    pred_np=(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
else:
    print('inference')
    pred = (model.inference(I0_p, I2_p, timestep=timestep, TTA=TTA, fast_TTA=TTA))
    pred = padder.unpad(pred)
    pred_np=(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

images = [I0[:, :, ::-1], pred_np[:, :, ::-1], I2[:, :, ::-1]]
mimsave('figs/out_2x.gif', images, fps=3)

pred = cv2.cvtColor(pred_np[:,:,::-1], cv2.COLOR_BGR2RGB)
psnr = cv2.PSNR(I1, pred)

#preds=preds.detach().cpu().numpy().transpose(1, 2, 0)*255
preds= pred_np[:,:,::-1]
cv2.imwrite( 'figs/preds.png',preds)

print(f'=========================Done=========================')
print(f'Model:{args.resume}  PSNR:{psnr}')

