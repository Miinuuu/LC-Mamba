from model.Residual_refiner import *
from model.Motion_estimator import *
from model.LC_Mamba_LFE_STFE import *

'''==========Model config=========='''
def init_model_config(F=32, 
                      in_chans=None,
                      embed_dims=None,
                      d_state=None,
                      W=None, 
                      depth=None,
                      expand=None,
                      scales=None,
                      hidden_dims=None,
                      refine=None,
                      window_shift=None,
                      ):
    '''This function should not be modified'''

    return { 
        'in_chans':in_chans or 3,
        'embed_dims':embed_dims or [F, 2*F, 4*F, 8*F, 16*F],
        'd_state': d_state or 16,
        'expand':expand or 2,
        'depths':depth or [2,2,2,2,2],
        'window_sizes':W or [8, 8],
        'window_shift': window_shift or 1,
    }, {
        'embed_dims': embed_dims or [F, 2*F, 4*F, 8*F, 16*F],
        'scales': scales or [8,16],
        'hidden_dims':hidden_dims or [4*F,4*F],
        'c':F,
        'refine': refine,
    }

def Model_create(model=None):
    if model in ['Ours-E'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (LC_Mamba_LFE_STFE, Motion_estimator),
        'MODEL_ARCH': init_model_config(
            F = 16,
            refine=Residual_refiner,
            W=[8,8],
            depth = [2,2,2,4,4])
        }
        
    elif model in ['Ours-B'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (LC_Mamba_LFE_STFE, Motion_estimator),
        'MODEL_ARCH': init_model_config(
            F = 32,
            refine=Residual_refiner,
            W=[8,8],
            depth = [2,2,2,2,2])
        }

    elif model in ['Ours-P'] :
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (LC_Mamba_LFE_STFE, Motion_estimator),
        'MODEL_ARCH': init_model_config(
            F = 32,
            refine=Residual_refiner,
            W=[16,16],
            window_shift=-1,
            depth = [2,2,2,4,4])
        }
    
    elif model in ['Ours-CS'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (LC_Mamba_LFE_STFE, Motion_estimator),
        'MODEL_ARCH': init_model_config(
            F = 16,
            refine=Residual_refiner,
            W=[8,8],
            depth = [2,2,2,2,2])
        }
    elif model in ['Ours-ES'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (LC_Mamba_LFE_STFE, Motion_estimator),
        'MODEL_ARCH': init_model_config(
            F = 16,
            refine=Residual_refiner,
            W=[8,8],
            depth = [2,2,2,4,4])
        }
        
    elif model in ['Ours-BS'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (LC_Mamba_LFE_STFE, Motion_estimator),
        'MODEL_ARCH': init_model_config(
            F = 32,
            refine=Residual_refiner,
            W=[8,8],
            depth = [2,2,2,2,2])
        }

    elif model in ['Ours-PS'] :
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (LC_Mamba_LFE_STFE, Motion_estimator),
        'MODEL_ARCH': init_model_config(
            F = 32,
            refine=Residual_refiner,
            W=[8,8],
            depth = [2,2,2,4,4])
        }
    
    return MODEL_CONFIG