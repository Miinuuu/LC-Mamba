import math
import numpy as np

def get_learning_rate(step,epoch_max,max_lr,min_lr,step_per_epoch,warmup_step=2000):
    if step < warmup_step:
        mul = step / warmup_step
        return max_lr * mul
    else:
        mul = np.cos((step - warmup_step) / (epoch_max * step_per_epoch - warmup_step) * math.pi) * 0.5 + 0.5
        return (max_lr - min_lr) * mul + min_lr

