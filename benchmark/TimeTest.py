from time import time
import sys
import torch
import argparse
import os
import warnings
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
from config import Model_create
from Trainer import Model

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Ours-B', type=str, help='Ours-C / Ours-E / Ours-B / Ours-P')
parser.add_argument('--resume', default=None, type=str, help='ckpt/<resume>.pkl (없으면 랜덤 가중치로 속도만 측정)')
parser.add_argument('--H', default=256, type=int)
parser.add_argument('--W', default=256, type=int)
args = parser.parse_args()

'''==========Model setting=========='''
model = Model(-1, Model_create(args.model))
if args.resume:
    model.load_checkpoint(args.resume)
model.eval()
model.device()

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

H, W = args.H, args.W
I0 = torch.rand(1, 3, H, W).cuda()
I1 = torch.rand(1, 3, H, W).cuda()
timestep = torch.tensor(0.5).reshape(1, 1, 1).unsqueeze(0).cuda()

print(f'Test model: {model.name}  TTA: {False}  size: {H}x{W}')
with torch.no_grad():
    for i in range(50):
        pred = model.inference(I0, I1, timestep=timestep)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_stamp = time()
    for i in range(100):
        pred = model.inference(I0, I1, timestep=timestep)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f'{(time() - time_stamp) / 100 * 1000:.2f} ms')
