import cv2
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from typing import List, Optional, Sequence, Union

#cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class vimeo_triplet(Dataset):
    def __init__(self, 
                split, 
                path,
                crop_size: Union[int, Sequence[int]] =None,
                resize=None):
        self.split = split
        self.data_root = path
        self.resize=resize
        self.crop_size = crop_size

        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()                                                    
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.split == 'train':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep=0.5
        return img0, gt, img1,timestep
            
    def __getitem__(self, index):        
        img0, gt, img1,timestep= self.getimg(index)
                
        if 'train' in self.split:
            if self.crop_size :
                h,w = self.crop_size
                img0, gt, img1 = self.aug(img0, gt, img1, h, w)

            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.resize :
            h,w=self.resize[0],self.resize[1]
            img0=cv2.resize(img0,(w,h) )
            gt=cv2.resize(gt,(w,h) )
            img1=cv2.resize(img1,(w,h) )

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)

        return torch.cat((img0, img1, gt), 0),timestep
