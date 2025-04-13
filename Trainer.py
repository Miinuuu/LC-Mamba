import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import os 
import re

############################################################################################################################
############################################################################################################################

class Model:
    def __init__(self, local_rank, 
                 MODEL_CONFIG):

        self.find_unused_parameters=MODEL_CONFIG['find_unused_parameters']
        self.name = MODEL_CONFIG['LOGNAME']
        print('Model:',self.name)

        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']            
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.net.to(torch.device("cuda"))


        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=self.find_unused_parameters)

        for param in self.net.parameters():
            param.requires_grad = True
            
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)

        param=sum(p.numel() for p in self.net.parameters())
    
        self.num_param = param/1000000
        print(f"Number of parameters: {self.num_param} M ")
       

    def device(self,):
        self.net.to(torch.device("cuda"))

    def eval(self,):
        self.net.eval()


    def convert(self, param):
            return {
            k.replace("module.", ""): v
                #for k, v in param['model_state_dict'].items()
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }

    def load_model(self, name=None, rank=0,training=False,strict_model=True):
            if rank <= 0 :
                if name is None:
                    name = self.name

                print('laod_model',name)

            if training :
                self.net.load_state_dict((torch.load(f'ckpt/{name}.pkl')),strict=strict_model)

            else:
                self.net.load_state_dict(self.convert(torch.load(f'ckpt/{name}.pkl')))
        
    def save_model(self, rank=0):
        if rank == 0:
            torch.save(self.net.state_dict(),f'ckpt/{self.name}.pkl')

    def load_checkpoint(self ,name=None,rank=0,training=False,strict_model=True):
        
        epoch=0
        global_step=0
        psnr=0
        #if rank <= 0 :
        if name is None:
            name = self.name
        print('---load_checkpoint---')
        print('resume : ',name)

        checkpoint = (torch.load(f'ckpt/{name}.pkl'))

        if training :                
            self.net.load_state_dict(checkpoint['model_state_dict'],strict=strict_model)
        else:
            self.net.load_state_dict(self.convert(checkpoint['model_state_dict']),strict=strict_model)
        
        self.optimG.load_state_dict(checkpoint['optimizer_state_dict'])
        
    
        psnr = checkpoint['psnr']
        name = checkpoint['name']

        print('model:',name)
        print('psnr:',psnr)

        return {'Model':name ,'epoch':epoch,'global_step':global_step,'psnr':psnr}

  
    def save_checkpoint(self,epoch,global_step,psnr,ssim,lpips,flolpips,rank=0):

        if rank == 0:
            checkpoint={
                'name':self.name,
                'psnr' : psnr,
                'ssim' : ssim,
                'lpips' : lpips,
                'flolpips' : flolpips,
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimG.state_dict(),
            }


            print('---save_checkpoint---')
            print('Model:',self.name)
            print('psnr:',psnr)
            print('epoch:',epoch)
            print('global_step:',global_step)

            torch.save(checkpoint,f'ckpt/{self.name}'+'_'+str(epoch)+'_'+str(psnr)[0:5]+'.pkl')
            checkpoint_files = [f for f in os.listdir('ckpt/') if  re.match(rf'^{re.escape(self.name)}_\d+', f) and f.endswith('.pkl')]
            checkpoint_files = sorted(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join('ckpt/', x)))
            print('checkpoint list:',checkpoint_files)

            save_ckt_n=4
            if len(checkpoint_files) > save_ckt_n:
                for file_to_delete in checkpoint_files[:-save_ckt_n]:
                    file_path = os.path.join('ckpt/', file_to_delete)
                    os.remove(file_path)
                    print(f"Remove checkpoint : {file_path}")


    @torch.no_grad()
    def inference(self, img0,img1=None, TTA = False, timestep = 0.5, fast_TTA = False):
        
        imgs = torch.cat((img0, img1), 1)
        '''
        Noting: return BxCxHxW
        '''
        _, _, _, pred = self.net(imgs,timestep=timestep)


        return pred
         
    @torch.no_grad()
    def hr_inference(self, img0, img1=None, TTA = False, down_scale = 1.0, timestep = 0.5, fast_TTA = False,local=False):
        '''
        Infer with down_scale flow
        Noting: return BxCxHxW
        '''
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)

            flow, mask = self.net.calculate_flow(imgs_down, timestep)

            flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
            mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)

            af= self.net.feature_bone(img0,img1)
      
            pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
            return pred

        imgs = torch.cat((img0, img1), 1)
        return infer(imgs)
      


    @torch.no_grad()
    def multi_inference(self, img0,img1, TTA = False, down_scale = 1.0, time_list=[], fast_TTA = False):
        '''
        Run backbone once, get multi frames at different timesteps
        Noting: return a list of [CxHxW]
        '''
        assert len(time_list) > 0, 'Time_list should not be empty!'
        af,mf=None,None
        afd,mfd=None,None
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            afmf = self.net.feature_bone(img0, img1)
            if isinstance(afmf, tuple):
                af,mf =afmf
            else:
                af=afmf 

            imgs_down = None
            if down_scale != 1.0:
                imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)
                afdmfd = self.net.feature_bone(imgs_down[:, :3], imgs_down[:, 3:6])

                if isinstance(afdmfd, tuple):
                    afd,mfd =afdmfd
                else:
                    afd=afdmfd 


            pred_list = []
            flow_list = []
            mask_list = []
            for timestep in time_list:
                if imgs_down is None:
                    flow, mask = self.net.calculate_flow(imgs, timestep, af, mf)
                else:
                    flow, mask = self.net.calculate_flow(imgs_down, timestep, afd, mfd)
                    flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
                    mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)
                
                pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
                pred_list.append(pred)
                flow_list.append(flow)
                mask_list.append(torch.sigmoid(mask))

            return pred_list
       

        imgs = torch.cat((img0, img1), 1)

        preds =  infer(imgs)


        return   [preds[i][0] for i in range(len(time_list))]

