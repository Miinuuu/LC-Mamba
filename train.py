import os
import math
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from Trainer import *
import hashlib
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid
from config import *
from model.scheduler import get_learning_rate
from model.visualize import flow2rgb_tensor,norm
import wandb as wandb
from typing import List, Optional, Sequence, Union
from tqdm import tqdm
from  datamodules.vimeo_triplet import vimeo_triplet
from  datamodules.vimeo_setuplet import vimeo_setuplet
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
#from model.softsplat import *
import inspect
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure
from fvcore.nn import flop_count_table
from fvcore.nn import FlopCountAnalysis
import time

device = torch.device("cuda")
exp = os.path.abspath('.').split('/')[-1]

def train(model, args ,epoch=0,global_step=0,val_psnr=0):
    
    val_ssim,val_lpips,val_flolpips=0,0,0

    start_epoch = epoch

    if args.dataset =='vimeo_triplet' :
        dataset = vimeo_triplet(split='train', path=args.data_path,crop_size=args.train_crop,resize=None)
        dataset_val = vimeo_triplet('test', args.data_path,crop_size=args.val_crop,resize=None)
    elif args.dataset =='vimeo_setuplet' :
        dataset = vimeo_setuplet(split='train', path=args.data_path,crop_size=args.train_crop,resize=None)
        dataset_val = vimeo_setuplet('test', args.data_path,crop_size=args.val_crop,resize=None)


    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=sampler)
    val_data = DataLoader(dataset_val, batch_size=1 , pin_memory=True, num_workers=1)
    step_per_epoch = train_data.__len__()

    print('Dataset:',args.dataset)
    print('step_per_epoch:',step_per_epoch)
    

    members = inspect.getmembers(model)
    for name, member in members:
        if inspect.ismethod(member) and name == args.updater:
            print('Updater:',args.updater)
            print('Trainer:',args.trainer)
            updater = member
            break


    flops_gflops=0.
    if args.flops_analysis :
        if args.local_rank == 0:
            with torch.no_grad():

                I0_p=torch.rand((1,3,256,256),device='cuda')
                I1_p=torch.rand((1,3,256,256),device='cuda')
                I2_p=torch.rand((1,3,256,256),device='cuda')
                timestep = torch.tensor(0.5).reshape(1, 1, 1).unsqueeze(0).cuda()
                flops = FlopCountAnalysis(model.net,(torch.cat((I0_p,I2_p),1),None,timestep))
                #flops = FlopCountAnalysis ( updater(torch.cat((I0_p,I2_p),1),I1_p,0,timestep,False),  (torch.cat((I0_p,I2_p),1),I1_p,0,timestep,False )  )   
            
            print(flop_count_table(flops))
            flops_gflops = flops.total() / 1e9

            print(f"Total FLOPs: {flops.total():,}")
            print(f"Total Gigaflops: {flops_gflops:.3f} GFLOPs")



    if args.first_val :
        if args.local_rank == 0:
            val_psnr,val_ssim,val_lpips,val_flolpips=evaluate(updater, val_data, epoch, args)
            '''model.save_checkpoint(epoch,
                            global_step,
                            val_psnr,
                            val_ssim,
                            val_lpips,
                            val_flolpips,
                            args.local_rank)'''    


    print('Training...')    
    for epoch in range(start_epoch,args.epoch_max) :
        with tqdm(train_data ) as  tepoch:
            sampler.set_epoch(epoch)
            for  (imgs,t) in ((tepoch)):
                imgs = imgs.to(device, non_blocking=True) / 255.
                imgs, gt = imgs[:, 0:6], imgs[:, 6:]
                learning_rate = get_learning_rate(step=global_step,
                                                epoch_max=args.epoch_max,
                                                step_per_epoch=step_per_epoch,
                                                max_lr=args.max_lr*(args.batch_size/32)*args.world_size,
                                                min_lr= args.min_lr or args.max_lr/10,
                                                warmup_step=args.warmup_step / args.world_size / (args.batch_size/32))

           
                pred, loss = updater(imgs=imgs, gt=gt, learning_rate=learning_rate, timestep=t, training=True)
                psnr = peak_signal_noise_ratio(pred, gt, data_range=1.0, dim=(1,2,3)).item()
                #ssim = structural_similarity_index_measure(pred, gt,data_range=1.0).item()

                if args.local_rank == 0:
                    tepoch.set_postfix({
                                        'Model' : args.model,
                                        'Epoch': epoch,
                                        'Params(M)':  model.num_param ,
                                        'FLOPs(G)': flops_gflops,
                                        'PSNR': psnr,
                                        'Val_PSNR':val_psnr,
                                        'Val_SSIM':val_ssim,
                                        'Loss': loss.item(), 
                                        'Lr':learning_rate,
                                        'Val_LPIPS':val_lpips,
                                        'Val_Flo_LPIPS':val_flolpips,
                                        }
                                        )
                    
                    run.log({
                            'epoch':epoch,
                            'loss':loss.item(),
                            'PSNR':psnr ,
                            #'SSIM':ssim,
                            'FLOPs(G)': flops_gflops,
                            'Params(M)':  model.num_param ,
                            'lr':learning_rate
                            })
                
                global_step += 1
            
            if (epoch % args.val_interval) == 0 or epoch == (args.epoch_max-1):
                if args.local_rank == 0:
                    val_psnr,val_ssim,val_lpips,val_flolpips=evaluate(updater, val_data, epoch, args)
                    model.save_checkpoint(epoch,
                                          global_step,
                                          val_psnr,
                                          val_ssim,
                                          val_lpips,
                                          val_flolpips,
                                          args.local_rank)    
                
            dist.barrier()
    if args.local_rank == 0:
        run.finish()


def evaluate(updater, val_data, epoch, args):
    print('Validating...')
    psnr = []
    ssim = []
    lpips = []
    flolpips=[]
    preds=[]
    gts=[]
    overlays=[]
    flows=[]
    masks=[]

    w_img_n= 20

    w_img_n_start=100
    w_img_n_end=w_img_n_start+w_img_n
    
    transform_to_tensor = transforms.ToTensor()
    transform_to_pil = transforms.ToPILImage()

    #font_size = 10
    #font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)  # Liberation Sans 폰트 사용
    #text_position = (10, 10)
        
    for i, (imgs,t) in enumerate(tqdm(val_data)):
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6].cuda(), imgs[:, 6:].cuda()
        t=t.cuda()

        with torch.no_grad():

            pred, flow ,mask, = updater(imgs=imgs, gt=gt, timestep=t,training=False)
            
            if i>= w_img_n_start and i < w_img_n_end :
                
                #if t.item() == 0.5 :
                '''else :
                    gt_t =transform_to_pil(gt[0])
                    draw = ImageDraw.Draw(gt_t)
                    draw.text(text_position, str(round(t.item(), 2)), font=font, fill="white")
                    gts.append(transform_to_tensor(gt_t).unsqueeze(0).cuda())'''

                
                gts.append(gt)
                preds.append(pred)
                masks.append(mask)
                flows.append(flow)#B,4,H,W
                overlays.append(t*imgs[:,0:3]+ (1-t)*imgs[:,3:6])



        psnr.append(peak_signal_noise_ratio(pred, gt, data_range=1.0, dim=(1,2,3)).item())
        ssim.append(structural_similarity_index_measure(pred, gt,data_range=1.0).item())
            
        if args.lpips : 
            lpips.append(lpips_fn(pred*2-1,gt*2-1).item())

        if args.flolpips :
            flolpips.append(flolpips_fn(imgs[:,0:3],imgs[:,3:6],pred,gt).item())

    psnr = np.array(psnr).mean()  
    ssim = np.array(ssim).mean()  

    if args.lpips:
        lpips = np.array(lpips).mean()  
    else:
        lpips=0.    

    if args.flolpips :
        flolpips = np.array(flolpips).mean()  
    else:
        flolpips=0.

    flows=torch.concat(flows,0) # B,4,H,W
    masks=torch.concat(masks,0) # B,1,H,W

    flowt0 = flows[:,0:2]
    flowt1 = flows[:,2:4]

    flowt0 = flow2rgb_tensor(flowt0)
    flowt1 = flow2rgb_tensor(flowt1)

    preds    = torch.concat(preds,0)
    overlays = torch.concat(overlays,0)
    gts      = torch.concat(gts,0)

    _,_,h,w=preds.shape
    

        
    preds = torch.cat([ overlays.unsqueeze(1),
                            gts.unsqueeze(1),
                            preds.unsqueeze(1),
                            torch.abs(preds-gts).unsqueeze(1),# pred diff error
                            flowt0.unsqueeze(1),
                            flowt1.unsqueeze(1),
                            masks.clip(0,1).repeat(1,3,1,1).unsqueeze(1),
                            ],dim=1)
    
    preds=(make_grid(preds.reshape(-1,3,h,w)[:,[2,1,0]], nrow=preds.shape[1],value_range=(0,1)))    
    
    log = { 'val_PSNR'  : psnr , 
            'val_SSIM'  : ssim,
            'epoch'     : epoch ,
            'preds'     : wandb.Image(preds.cpu(), file_type="jpg",mode='RGB',caption='preds')}

    if args.lpips : 
        log.update({'val_LPIPS': lpips})
 

    if args.flolpips :
        log.update({'val_FloLPIPS': flolpips})


    if args.local_rank == 0:
        run.log(log)
        print('epoch:{} psnr:{:.2f} dB ssim:{:.3f} lpips:{:.3f}  flolpips : {:.3f}'.format(epoch,psnr,ssim,lpips,flolpips))
    
    del overlays
    del gts
    del preds
    del flowt0
    del flowt1
    del masks
    torch.cuda.empty_cache()

    return psnr,ssim,lpips,flolpips

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model'           ,default=None                               ,type=str                       ,help='model')
    parser.add_argument('--trainer'         ,default='Model'                            ,type=str                       ,help='trainer')
    parser.add_argument('--updater'         ,default='update'                           ,type=str                       ,help='updater')
    parser.add_argument('--resume'          ,default=None                               ,type=str                       ,help='resume')
    parser.add_argument('--local-rank'      ,default=0                                  ,type=int                       ,help='local rank')
    parser.add_argument('--world_size'      ,default=1                                  ,type=int                       ,help='world size')
    parser.add_argument('--num_workers'     ,default=16                                 ,type=int                       ,help='num_workers ')
    parser.add_argument('--batch_size'      ,default=32                                 ,type=int                       ,help='batch size')
    parser.add_argument('--epoch_max'       ,default=300                                ,type=int                       ,help='epoch size')
    parser.add_argument('--max_lr'          ,default=2e-4                               ,type=float                     ,help='max learning rate')
    parser.add_argument('--min_lr'          ,default=None                               ,type=float                     ,help='min learning rate')
    parser.add_argument('--warmup_step'     ,default=2000                               ,type=int                       ,help='warmup step')
    parser.add_argument('--train_crop'      ,default=[256,256]                          ,type=Union[int, Sequence[int]] ,help='train crop size')
    parser.add_argument('--val_interval'    ,default=5                                  ,type=int                       ,help='validation interval')
    parser.add_argument('--val_crop'        ,default=None                               ,type=Union[int, Sequence[int]] ,help='val crop size')
    parser.add_argument('--data_path'       ,default='/data/datasets/vimeo_triplet'     ,type=str                       ,help='data path of dataset')
    parser.add_argument('--dataset'         ,default='vimeo_triplet'                    ,type=str                       ,help='tpye of dataset')
    parser.add_argument('--epoch_start'     ,default=None                               ,type=int                       ,help='epoch_start')
    parser.add_argument('--global_step'     ,default=None                               ,type=int                       ,help='global_step')
    parser.add_argument('--project'         ,default='my'                               ,type=str                       ,help='wandb project name')
    parser.add_argument('--first_val'       ,default=False                              ,action='store_true'            ,help='first validation')
    parser.add_argument('--strict_model'    ,default=False                               ,action='store_true'            ,help='strict model')
    parser.add_argument('--lpips'           ,default=False                              ,action='store_true'            ,help='lpips_metric')
    parser.add_argument('--flolpips'        ,default=False                              ,action='store_true'            ,help='flolpips_metric')
    parser.add_argument('--flops_analysis'  ,default=True                               ,action='store_true'            ,help='flops_analysis')

    args = parser.parse_args()

    print('argument:',args)
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    MODEL_CONFIG= Model_create(args.model)
    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')



    
    exec(args.trainer, globals())
    created_class = globals()[args.trainer]


    if args.lpips : 
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').cuda().eval() #alex, vgg,squeeze

    if args.flolpips :
        from flolpips.flolpips import Flolpips
        flolpips_fn = Flolpips().cuda().eval()

    if args.resume :
        print('Resume training:',args.resume)
        #model = Model(args.local_rank,MODEL_CONFIG)
        model = created_class(local_rank=args.local_rank, MODEL_CONFIG=MODEL_CONFIG)
        ckpt=model.load_checkpoint(args.resume,args.local_rank,training=True,strict_model=args.strict_model)
        epoch=ckpt['epoch'] if args.epoch_start == None else 0
        global_step=ckpt['global_step'] if args.global_step == None else 0
        val_psnr=ckpt['psnr']

        if args.strict_model :
            assert args.model == ckpt['Model']
        
        note = str(model.num_param) +'M'

        if args.local_rank == 0:
            wandb.login()
            log_dir = Path.cwd().absolute() / "wandb_logs" / args.model
            log_dir.mkdir(exist_ok=True, parents=True)
            sha = hashlib.sha256()
            sha.update(str(args.model).encode())
            wandb_id = sha.hexdigest()

            run = wandb.init(project=args.project,
                            id=  wandb_id, 
                            dir= log_dir,
                            job_type='train',
                            save_code=True,
                            notes=note,
                            name=args.model,
                            resume='allow')
        train(model, args,epoch+1,global_step,val_psnr)

    else:
        print('New training:',args.model)
        model = created_class(local_rank=args.local_rank,MODEL_CONFIG=MODEL_CONFIG)
        
        note = str(model.num_param) +'M'

        if args.local_rank == 0:
            wandb.login()
            log_dir = Path.cwd().absolute() / "wandb_logs" / args.model
            log_dir.mkdir(exist_ok=True, parents=True)
            sha = hashlib.sha256()
            sha.update(str(args.model).encode())
            wandb_id = sha.hexdigest()

            run = wandb.init(project=args.project,
                            id=  wandb_id, 
                            dir= log_dir,
                            job_type='train',
                            save_code=True,
                            notes=note,
                            name=args.model,
                            resume='allow')
        
        #model = Model(args.local_rank,MODEL_CONFIG)
        #Model.load_model('ours_t',args.local_rank,True,False)
        #model.save_checkpoint(299,0,31.9,0.97,1.0,1.0,args.local_rank)    
        
        train(model, args)
