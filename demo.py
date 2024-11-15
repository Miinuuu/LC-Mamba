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
from fvcore.nn import flop_count_table
from fvcore.nn import FlopCountAnalysis
import wandb as wandb
import hashlib
from pathlib import Path
from torchvision.utils import make_grid
from model.visualize import flow2rgb_tensor,norm


parser = argparse.ArgumentParser()


parser.add_argument('--model', default='lite', type=str)
parser.add_argument('--resume', default=None, type=str, help='resume')
parser.add_argument('--trainer', type=str, default='Model',help='trainer')
parser.add_argument('--size', type=Union[int, Sequence[int]], default=[4096,2048],help='img size')
parser.add_argument('--scale', type=int, default=0.25,help='img scale')
parser.add_argument('--flops_analysis', action='store_true',default=False,help='flops_analysis')
parser.add_argument('--strict_model', action='store_true', default=False,help='strict model')
parser.add_argument('--project', type=str, default='demo',help='wandb project name')

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


wandb.login()
log_dir = Path.cwd().absolute() / "wandb_logs" / args.model
log_dir.mkdir(exist_ok=True, parents=True)
sha = hashlib.sha256()
sha.update(str(args.model).encode())
wandb_id = sha.hexdigest()

note = str(model.num_param) +'M'
run = wandb.init(project=args.project,
                id=  wandb_id, 
                dir= log_dir,
                job_type='demo',
                save_code=True,
                notes=note,
                name=args.model,
                resume='allow')


w,h = args.size[0],args.size[1]
down_scale=args.scale
print(f'=========================Start Generating=========================')

I0 = cv2.imread('example/Beanbags/frame07.png')
I1 = cv2.imread('example/Beanbags/frame08.png')
I2 = cv2.imread('example/Beanbags/frame09.png')

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

if args.flops_analysis :
        flops = FlopCountAnalysis(model.net,(torch.cat((I0_p,I2_p),1),None,timestep))
        print(flop_count_table(flops))
        flops_gflops = flops.total() / 1e9

        print(f"Total FLOPs: {flops.total():,}")
        print(f"Total Gigaflops: {flops_gflops:.3f} GFLOPs")

else :
    if down_scale != 1.0 :
        print('hr inference')
        pred,flow,mask= (model.hr_inference(I0_p,None, I2_p, timestep=timestep,TTA=TTA, down_scale=down_scale,fast_TTA=TTA))
        pred = padder.unpad(pred)
        pred_np=(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    else:
        print('inference')
        pred,flow,mask= (model.inference(I0_p, None,I2_p, timestep=timestep, TTA=TTA, fast_TTA=TTA))
        pred = padder.unpad(pred)
        pred_np=(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)


    flowt0 = flow2rgb_tensor(flow[:,0:2])
    flowt0 = padder.unpad(flowt0)

    flowt1 = flow2rgb_tensor(flow[:,2:4])
    flowt1 = padder.unpad(flowt1)
    mask = padder.unpad(mask)

    AD =  torch.abs(pred-I1_)

    preds=torch.cat([(timestep*I0_+(1-timestep)*I2_),
                     I1_,
                     pred,
                     AD,
                     flowt0,
                     flowt1,
                     mask.repeat(1,3,1,1),
                     ] ,dim=0) # overlay, gt, pred
    
    preds=(make_grid(preds[:,[2,1,0]], nrow=preds.shape[1],value_range=(0,1)))    

    images = [I0[:, :, ::-1], pred_np[:, :, ::-1], I2[:, :, ::-1]]
    mimsave('example/out_2x.gif', images, fps=3)

    pred = cv2.cvtColor(pred_np[:,:,::-1], cv2.COLOR_BGR2RGB)
    psnr = cv2.PSNR(I1, pred)

    preds=preds.detach().cpu().numpy().transpose(1, 2, 0)*255
    preds= preds[:,:,::-1]
    cv2.imwrite( f'example/{args.resume}_{args.size}_{args.scale}_preds.png',preds)

    '''cv2.imwrite( f'example/{args.resume}_{args.size}_{args.scale}_pred.png',pred)
    cv2.imwrite( f'example/{args.resume}_{args.size}_{args.scale}_gt.png',I1)
    cv2.imwrite( f'example/{args.resume}_{args.size}_{args.scale}_overlay.png',(0.5*I0+0.5*I2))
    cv2.imwrite( f'example/{args.resume}_{args.size}_{args.scale}_AD.png',abs(I1-pred))
    '''
    print(f'=========================Done=========================')
    print(f'Model:{args.resume}  PSNR:{psnr}')

    run.log({   'demo_PSNR' : psnr , 
                'epoch' : 0 ,
                'demo_preds': wandb.Image(preds, file_type="jpg",mode='RGB',caption='demo_pred_'+str(args.size)+'_'+str(args.scale)),
                #'pred': wandb.Image(mid[:, :, ::-1], file_type="jpg",mode='RGB',caption='demo_pred_'+str(args.size)+'_'+str(args.scale)),
                #'overlay': wandb.Image(0.5*I0[:, :, ::-1]+0.5*I1[:, :, ::-1], file_type="jpg",mode='RGB',caption='demo_overlay_'+str(args.size)+'_'+str(args.scale)),
                #'gt': wandb.Image(gt[:,:,::-1], file_type="jpg",mode='RGB',caption='demo_gt_'+str(args.size)+'_'+str(args.scale)) ,
                })

run.finish