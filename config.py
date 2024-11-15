from functools import partial
import torch.nn as nn
from model.feature_extractor import * 
from model.refine import *
from model.flow_estimation import *
from model.flow_estimation2 import *

from model.feature_extractor_mamba import * 
from model.flow_estimation_mamba import *
#import model_vmamba.feature_extractor as feature_extractor_vmamba
#import model_vmamba.flow_estimation as flow_estimation_vmamba

'''==========Model config=========='''
def init_model_config(F=32, 
                      embed_dims=None,
                      motion_dims=None,
                      W=None, 
                      in_chans=None,
                      depth=None,
                      mlp_ratios=None,
                      num_heads=None,
                      scales=None,
                      hidden_dims=None,
                      refine=None,
                      motion=True,
                      atime=False):
    '''This function should not be modified'''

    return { 
        'embed_dims':embed_dims or [F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims':motion_dims or [0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads': num_heads or [8*F//32, 16*F//32],
        'mlp_ratios':mlp_ratios or [4, 4],
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6), 
        'depths':depth or [2,2,2,2,2],
        'window_sizes':W or [7, 7],
        'in_chans':in_chans or 3
    }, {
        'embed_dims': embed_dims or [F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims' :motion_dims or [0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths':depth or [2, 2, 2, 2, 2],
        'num_heads':num_heads or [8*F//32, 16*F//32],
        'window_sizes': W or [7, 7],
        'scales': scales or [8,16],
        'hidden_dims':hidden_dims or [4*F,4*F],
        'c':F,
        'refine': refine,
        'motion': motion,
        'atime' : atime,
    }

'''==========Model config=========='''
def init_model_config_mamba(F=32, W=7, depth=[2, 2, 2, 4, 4], M=False):
    '''This function should not be modified'''
    return { 
        'embed_dims':[(2**i)*F for i in range(len(depth))],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads':[8*(2**i)*F//32 for i in range(len(depth)-3)],
        'mlp_ratios':[4 for i in range(len(depth)-3)],
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6), 
        'depths':depth,
        'window_sizes':[W for i in range(len(depth)-3)],
        'conv_stages':3
    }, {
        'embed_dims':[(2**i)*F for i in range(len(depth))],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths':depth,
        'num_heads':[8*(2**i)*F//32 for i in range(len(depth)-3)],
        'window_sizes':[W, W],
        'scales':[4*(2**i) for i in range(len(depth)-2)],
        'hidden_dims':[4*F for i in range(len(depth)-3)],
        'c':F,
        'M':M,
        'local_hidden_dims':4*F,
        'local_num':2
    }

def Model_create(model=None):
    note=''

    if model in ['ours_small' ,'ours_small_t', 'my_t', 'my','small']:
        F=16
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor, MultiScaleFlow),
        'MODEL_ARCH': init_model_config(
            F = 16,
            W=[7,7],
            depth = [2, 2, 2, 2, 2],
            refine=Unet)
        ,'DISTILL' : None
        ,'BASE':None
        }


            
    elif model in ['ours','ours_t'] :
        
        F=32
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor, MultiScaleFlow),
        'MODEL_ARCH': init_model_config(
            F = 32,
            W=[7,7],
            depth = [2, 2, 2, 4, 4]
            ,refine=Unet)
        ,'DISTILL' : None
        ,'BASE':None

        }
    elif model in ['small_nm_nrf'] :
        
        F=16
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor_nm, MultiScaleFlow),
        'MODEL_ARCH': init_model_config(
            F = 16,
            W=[7,7],
            depth = [2, 2, 2, 2, 2]
            ,refine=None
            ,motion=False
            )
        ,'DISTILL' : None
        ,'BASE':None

        }
    elif model in ['small_nm_nrf_r22'] :
        
        F=16
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor_nm, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            W=[7,7],
            depth = [2, 2, 2, 2, 2]
            ,refine=r22
            ,motion=False
            )
        ,'DISTILL' : None
        ,'BASE':None

        }
    elif model in ['small_nm_nrf_step2'] :
        
        F=16
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPEB': (feature_extractor_nm, MultiScaleFlow),
        'MODEL_ARCHB': init_model_config(
            F = 16,
            W=[7,7],
            depth = [2, 2, 2, 2, 2]
            ,refine=None
            ,motion=False
            ),
        'MODEL_TYPE': (feature_extractor_nm, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            W=[7,7],
            depth = [2, 2, 2, 2, 2]
            ,refine=r2
            ,motion=False
            )
        ,'DISTILL' : None
        ,'BASE':'small_nm_nrf_299_35.17'

        }
    elif model in ['small_nm_tiny'] :
        
        F=16
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor_lite_nm, MultiScaleFlow),
        'MODEL_ARCH': init_model_config(
            F = 16,
            W=[7,7],
            depth = [2, 2, 2, 2, 2]
            ,refine=Unet_srf1
            ,motion=False
            )
        ,'DISTILL' : None
        ,'BASE':None

        }

 
    elif model in ['lite'] :
        
        F=16
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor_lite, MultiScaleFlow),
        'MODEL_ARCH': init_model_config(
            F = 16,
            W=[7,7],
            depth = [2, 2, 2, 2, 2]
            ,refine=Unet)
        ,'DISTILL' : None
        ,'BASE':None
        }
        note='lite'




    elif model in ['my_t_nrf']:

        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor, MultiScaleFlow),
        'MODEL_ARCH': init_model_config(
            F = 16,
            W=[7,7],
            depth = [2, 2, 2, 2, 2],
            refine=None,
            atime=True,
            )
        }
        note='my_t_nrf'
        
    elif model in ['lite_nrf']:
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor_lite, MultiScaleFlow),
        'MODEL_ARCH': init_model_config(
            F = 16,
            W=[7,7],
            depth = [2, 2, 2, 2, 2],
            refine=None)
        ,'DISTILL' : None
        ,'BASE':None
        }
        note='lite_no_refine'
        
    elif model in ['lite_nm'] :
        
        F=16
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor_lite_nm, MultiScaleFlow),
        'MODEL_ARCH': init_model_config(
            F = 16,
            W=[7,7],
            depth = [2, 2, 2, 2, 2],
            refine=None
            ,motion=False)
        ,'DISTILL' : None
        ,'BASE':None
        }
        note='lite_nm'
    

    elif model in ['nm_srf']:
        
        F=16
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor_nm, MultiScaleFlow),
        'MODEL_ARCH': init_model_config(
            F = F,
            W = [7,7],
            depth = [2, 2, 2, 2, 2],
            motion=False,
            refine=Unet_srf2
            )  ,
        'DISTILL':None
        ,'BASE':None
        }


    elif model in ['big_srf']:
        
        F=32
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor_nm, MultiScaleFlow),
        'MODEL_ARCH': init_model_config(
            F = F,
            W=[7,7],
            depth = [2, 2, 2, 2, 2],
            motion=False,
            refine=Unet_srf2
            )  ,
        'DISTILL':None
        ,'BASE':None

        }


            
            
    elif model in ['csdw_dec_dq'] :
            
            F=16
            MODEL_CONFIG = {
            'LOGNAME': model ,
        'find_unused_parameters':False,
            'MODEL_TYPE': (feature_extractor_nm_csdw, MultiScaleFlow_dec_opt),
            'MODEL_ARCH': init_model_config(
                F = 16,
                W=[7,7],
                depth = [2, 2, 2, 2, 2],
                hidden_dims=[4*F,4*F]
                ,refine=dec_dq
                ,motion=False
                )
            ,'DISTILL' : None
            ,'BASE':None
            }
    elif model in ['csdw_r2_step2'] :
            
            F=16
            MODEL_CONFIG = {
            'LOGNAME': model ,
        'find_unused_parameters':False,
            'MODEL_TYPEB': (feature_extractor_nm_csdw, MultiScaleFlow_dec_opt),
            'MODEL_ARCHB': init_model_config(
                F = 16,
                W=[7,7],
                depth = [2, 2, 2, 2, 2],
                hidden_dims=[4*F,4*F]
                ,refine=dec_dq
                ,motion=False
                ),
            'MODEL_TYPE': (feature_extractor_nm_csdw, MultiScaleFlow_dec_opt),
            'MODEL_ARCH': init_model_config(
                F = 16,
                W=[7,7],
                depth = [2, 2, 2, 2, 2],
                hidden_dims=[4*F,4*F]
                ,refine=r2
                ,motion=False
                )
            ,'DISTILL' : None
            ,'BASE':'csdw_dec_dq_299_36.14'
            }
            
            
    elif model in ['csdw_dec_dq_opt'] :
            
            F=16
            MODEL_CONFIG = {
            'LOGNAME': model ,
        'find_unused_parameters':False,
            'MODEL_TYPE': (feature_extractor_nm_csdw, MultiScaleFlow_dec_opt),
            'MODEL_ARCH': init_model_config(
                F = 16,
                W=[7,7],
                depth = [2, 2, 2, 2, 2],
                hidden_dims=[4*F,4*F]
                ,refine=dec_dq_opt
                ,motion=False
                )
            ,'DISTILL' : None
            ,'BASE':None
            }
    
            
    elif model in ['ov_dec_dq_opt'] :
            
            F=16
            MODEL_CONFIG = {
            'LOGNAME': model ,
        'find_unused_parameters':False,
            'MODEL_TYPE': (MotionFormer_ov_nm, MultiScaleFlow_dec_opt),
            'MODEL_ARCH': init_model_config(
                F = 16,
                W=[7,7],
                depth = [2, 2, 2, 2, 2],
                hidden_dims=[4*F,4*F]
                ,refine=dec_dq_opt
                ,motion=False
                )
            ,'DISTILL' : None
            ,'BASE':None
            }
    elif model in ['ov_dec_r2_step2'] :
            
            F=16
            MODEL_CONFIG = {
            'LOGNAME': model ,
        'find_unused_parameters':False,
            'MODEL_TYPEB': (MotionFormer_ov_nm, MultiScaleFlow_dec_opt),
            'MODEL_ARCHB': init_model_config(
                F = 16,
                W=[7,7],
                depth = [2, 2, 2, 2, 2],
                hidden_dims=[4*F,4*F]
                ,refine=dec_dq_opt
                ,motion=False
                ),
            'MODEL_TYPE': (MotionFormer_ov_nm, MultiScaleFlow_dec_opt),
            'MODEL_ARCH': init_model_config(
                F = 16,
                W=[7,7],
                depth = [2, 2, 2, 2, 2],
                hidden_dims=[4*F,4*F]
                ,refine=r2
                ,motion=False
                )
            ,'DISTILL' : None
            ,'BASE':'ov_dec_dq_opt_105_35.72'
            }
    
            
    elif model in ['ov_step1'] :
            
            F=16
            MODEL_CONFIG = {
            'LOGNAME': model ,
        'find_unused_parameters':False,
            'MODEL_TYPE': (MotionFormer_ov_nm, MultiScaleFlow_dec_opt),
            'MODEL_ARCH': init_model_config(
                F = 16,
                W=[7,7],
                depth = [2, 2, 2, 2, 2],
                hidden_dims=[4*F,4*F]
                ,refine=None
                ,motion=False
                )
            ,'DISTILL' : None
            ,'BASE':None
            }
    elif model in ['ov_step2'] :
            
            F=16
            MODEL_CONFIG = {
            'LOGNAME': model ,
        'find_unused_parameters':False,
            'MODEL_TYPEB': (MotionFormer_ov_nm, MultiScaleFlow_dec_opt),
            'MODEL_ARCHB': init_model_config(
                F = 16,
                W=[7,7],
                depth = [2, 2, 2, 2, 2],
                hidden_dims=[4*F,4*F]
                ,refine=None
                ,motion=False
                ),
            'MODEL_TYPE': (MotionFormer_ov_nm, MultiScaleFlow_dec_opt),
            'MODEL_ARCH': init_model_config(
                F = 16,
                W=[7,7],
                depth = [2, 2, 2, 2, 2],
                hidden_dims=[4*F,4*F]
                ,refine=r2
                ,motion=False
                )
            ,'DISTILL' : None
            ,'BASE':'ov_step1_299_35.23'
            }
    elif model in ['ov_dec_dq'] :
            
            F=16
            MODEL_CONFIG = {
            'LOGNAME': model ,
        'find_unused_parameters':False,
            'MODEL_TYPE': (MotionFormer_ov_nm, MultiScaleFlow_dec_opt),
            'MODEL_ARCH': init_model_config(
                F = 16,
                W=[7,7],
                depth = [2, 2, 2, 2, 2],
                hidden_dims=[4*F,4*F]
                ,refine=dec_dq
                ,motion=False
                )
            ,'DISTILL' : None
            ,'BASE':None
            }
    
    

    elif model in ['zorder3_dec_dq_opt22'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            W = 7,
            motion=False,
            refine=dec_dq_opt,
            depth = [2, 2, 2, 2, 2])
        ,'DISTILL' : None
        ,'BASE':None
        }
  
    elif model in ['zorder3_dec_dq_opt'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            W = 7,
            motion=False,
            refine=dec_dq_opt,
            depth = [2, 2, 2, 3, 3])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['cross_dec_dq_opt'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_cross, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            W = 7,
            motion=False,
            refine=dec_dq_opt,
            depth = [2, 2, 2, 3, 3])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['zorder_local_cross_dec_dq_opt'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_cross, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2, 2, 2, 3, 3])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['local_cross_dec_dq_opt'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_local_cross, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
           
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2, 2, 2, 3, 3])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['local_dec_dp_opt'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_local, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2, 2, 2, 3, 3])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['zorder_local_dec_dp_opt'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2, 2, 2, 3, 3])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['zorder_local_dec_dp_opt_big'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2, 2, 2, 2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }
        
    elif model in ['zorder_local_shift_dec_dq_opt_big'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2, 2, 2, 2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['zorder_local_shift_dec_dq_opt_1633'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,3,3])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['MotionMamba2_local_shift_dec_dq_opt_1633'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba2_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,3,3])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['zorder_local_shift_dec_dq_opt_3222'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['zorder_nrf_local_shift_1622_step1'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }
        

    elif model in ['zorder_nrf_local_shift_1622_step2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPEB': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCHB': init_model_config(
            F = 16,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2]),
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=r3,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':'zorder_nrf_local_shift_1622_step1_140_34.98'
        }

    elif model in ['zorder_nrf_local_shift_1644_step1'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
  
    elif model in ['zorder_nrf_local_shift_1688_step1'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,8,8])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['zorder_nrf_local_shift_2422_step1'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 24,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['zorder_nrf_local_shift_2444_step1'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 24,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['zorder_nrf_local_shift_2488_step1'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 24,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,8,8])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['v2_nrf_local_shift_s_1622_step1'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':True,
        'MODEL_TYPE': (MotionMamba_v2_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['v2_nrf_local_1622_step1'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_v2_local, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2])

        ,'DISTILL' : None
        ,'BASE':None
    }
    elif model in ['v2_nrf_local_1622_step2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPEB': (MotionMamba_v2_local, MultiScaleFlow_dec_opt),
        'MODEL_ARCHB': init_model_config(
            F = 16,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2]),
        'MODEL_TYPE': (MotionMamba_v2_local, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=r2,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':'v2_nrf_local_1622_step1_290_35.19'
    }
    elif model in ['v2_nrf_local_1622_r3_step2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPEB': (MotionMamba_v2_local, MultiScaleFlow_dec_opt),
        'MODEL_ARCHB': init_model_config(
            F = 16,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2]),
        'MODEL_TYPE': (MotionMamba_v2_local, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=r3,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':'v2_nrf_local_1622_step1_290_35.19'
    }
        
    elif model in ['zorder_nrf_local_shift_3222_step1'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }
    
    elif model in ['zorder_nrf_local_shift_3244_step1'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
    
    elif model in ['zorder_nrf_local_shift_4822_step1'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 48,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['zorder_nrf_local_shift_4844_step1'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 48,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }


    elif model in ['zorder_nrf_local_shift_1622_r2_step2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPEB': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCHB': init_model_config(
            F = 16,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2]),
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=r2,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':'zorder_nrf_local_shift_1622_step1_299_35.18'
        }
    elif model in ['zorder_nrf_local_shift_1622_r3_step2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPEB': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCHB': init_model_config(
            F = 16,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2]),
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=r3,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':'zorder_nrf_local_shift_1622_step1_299_35.18'
        }

    elif model in ['zorder_nrf_local_shift_3222_r_step2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':True,
        'MODEL_TYPEB': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCHB': init_model_config(
            F = 32,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2]),
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=r,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':'zorder_nrf_local_shift_3222_step1_299_35.48'
        }

    elif model in ['zorder_nrf_local_shift_3222_r2_step2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPEB': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCHB': init_model_config(
            F = 32,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2]),
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=r2,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':'zorder_nrf_local_shift_3222_step1_299_35.48'
        }
 
        
    elif model in ['zorder_nrf_local_shift_3222_r2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=r2,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE': None
        }

    elif model in ['zorder_nrf_local_shift_3222_r3_step2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':True,
        'MODEL_TYPEB': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCHB': init_model_config(
            F = 32,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2]),
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=r3,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':'zorder_nrf_local_shift_3222_step1_299_35.48'
        }
    elif model in ['zorder_nrf_local_shift_3222_r4_step2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':True,
        'MODEL_TYPEB': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCHB': init_model_config(
            F = 32,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2]),
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=r4,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':'zorder_nrf_local_shift_3222_step1_299_35.48'
        }
    elif model in ['zorder_nrf_local_shift_3222_r5_step2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':True,
        'MODEL_TYPEB': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCHB': init_model_config(
            F = 32,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2]),
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=r5,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':'zorder_nrf_local_shift_3222_step1_299_35.48'
        }
    elif model in ['zorder_nrf_local_shift_3222_dec_dq_opt_step2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPEB': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCHB': init_model_config(
            F = 32,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2]),
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':'zorder_nrf_local_shift_3222_step1_299_35.48'
        }
    elif model in ['zorder_nrf_local_shift_3222_unet_step2'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPEB': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCHB': init_model_config(
            F = 32,
            motion=False,
            refine=None,
            W=[8,8],
            depth = [2,2,2,2,2]),
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=Unet,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':'zorder_nrf_local_shift_3222_step1_299_35.48'
        }


    elif model in ['zorder_local_shift_dec_dq_opt_big64'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_zorder_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 64,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }


#############2d###########################2d###########################2d###########################2d###########################2d###########################2d##############
#############2d###########################2d###########################2d###########################2d###########################2d###########################2d##############
#############2d###########################2d###########################2d###########################2d###########################2d###########################2d##############


    elif model in ['hilbert_local_shift_dec_dq_opt_1633'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,3,3])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['hilbert_local_shift_dec_dq_opt_1644'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['hilbert_local_shift_dec_dq_opt_1644_wos'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2_wos, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['hilbert_local_shift_dec_dq_opt_1644_w4'] :
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[4,4],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['hilbert_local_shift_dec_dq_opt_1644_w4_wos'] :
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2_wos, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[4,4],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['hilbert_local_shift_dec_dq_opt_1644_w16'] :
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[16,16],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['hilbert_local_shift_dec_dq_opt_1644_w16_wos'] :
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2_wos, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[16,16],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
    


    elif model in ['hilbert_local_shift_dec_dq_opt_3222'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['hilbert_local_shift_dec_dq_opt_3222_wos'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2_wos, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['hilbert_local_shift_dec_dq_opt_3222_w16_wos'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2_wos, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[16,16],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['hilbert_local_shift_dec_dq_opt_3222_w16'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[16,16],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['hilbert_local_shift_dec_dq_opt_3222_t'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,2,2])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['hilbert_local_shift_dec_dq_opt_3244'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
        
    elif model in ['hilbert_local_shift_dec_dq_opt_3244_wos'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2_wos, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
        
    elif model in ['hilbert_local_shift_dec_dq_opt_3244_w16'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[16,16],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
        
    elif model in ['hilbert_local_shift_dec_dq_opt_3244_w16_wos'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2_wos, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[16,16],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['hilbert_local_shift_dec_dq_opt_3244_t'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['hilbert_local_shift_dec_dq_opt_3288'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,8,8])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['hilbert_local_shift_dec_dq_opt_3288_w16_wos'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2_wos, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[16,16],
            depth = [2,2,2,8,8])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['hilbert_local_shift_dec_dq_opt_3266'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_local_shift2, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 32,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,6,6])
        ,'DISTILL' : None
        ,'BASE':None
        }


    # elif model in ['hilbert_local_shift_rot_dec_dq_opt_1633'] :
        
    #     MODEL_CONFIG = {
    #     'LOGNAME': model ,
    #     'find_unused_parameters':False,
    #     'MODEL_TYPE': (MotionMamba_hilbert_local_shift_rot, MultiScaleFlow_dec_opt),
    #     'MODEL_ARCH': init_model_config(
    #         F = 16,
    #         motion=False,
    #         refine=dec_dq_opt,
    #         W=[8,8],
    #         depth = [2,2,2,3,3])
    #     ,'DISTILL' : None
    #     ,'BASE':None
    #     }
    # elif model in ['hilbert_local_shift_rot_dec_dq_opt_1644'] :
        
    #     MODEL_CONFIG = {
    #     'LOGNAME': model ,
    #     'find_unused_parameters':False,
    #     'MODEL_TYPE': (MotionMamba_hilbert_local_shift_rot, MultiScaleFlow_dec_opt),
    #     'MODEL_ARCH': init_model_config(
    #         F = 16,
    #         motion=False,
    #         refine=dec_dq_opt,
    #         W=[8,8],
    #         depth = [2,2,2,4,4])
    #     ,'DISTILL' : None
    #     ,'BASE':None
    #     }
    # elif model in ['hilbert_local_shift_rot_dec_dq_opt_3222'] :
        
    #     MODEL_CONFIG = {
    #     'LOGNAME': model ,
    #     'find_unused_parameters':False,
    #     'MODEL_TYPE': (MotionMamba_hilbert_local_shift_rot, MultiScaleFlow_dec_opt),
    #     'MODEL_ARCH': init_model_config(
    #         F = 32,
    #         motion=False,
    #         refine=dec_dq_opt,
    #         W=[8,8],
    #         depth = [2,2,2,2,2])
    #     ,'DISTILL' : None
    #     ,'BASE':None
    #     }
    # elif model in ['hilbert_local_shift_rot_dec_dq_opt_3244'] :
        
    #     MODEL_CONFIG = {
    #     'LOGNAME': model ,
    #     'find_unused_parameters':False,
    #     'MODEL_TYPE': (MotionMamba_hilbert_local_shift_rot, MultiScaleFlow_dec_opt),
    #     'MODEL_ARCH': init_model_config(
    #         F = 32,
    #         motion=False,
    #         refine=dec_dq_opt,
    #         W=[8,8],
    #         depth = [2,2,2,4,4])
    #     ,'DISTILL' : None
    #     ,'BASE':None
    #     }
    # elif model in ['hilbert_local_shift_rot_dec_dq_opt_16222'] :
    #     F=16
    #     MODEL_CONFIG = {
    #     'LOGNAME': model ,
    #     'find_unused_parameters':False,
    #     'num_heads': [8*F//32, 16*F//32,32*F//32],
    #     'MODEL_TYPE': (MotionMamba_hilbert_local_shift_rot, MultiScaleFlow_dec_opt),
    #     'MODEL_ARCH': init_model_config(
    #         F = 16,
    #         motion=False,
    #         refine=dec_dq_opt,
    #         W=[8,8],
    #         embed_dims= [F, 2*F, 4*F, 8*F, 16*F ,32*F],
    #         scales =[8,16,32],
    #         hidden_dims= [4*F,4*F,4*F],
    #         depth = [2,2,2,2,2,2])
    #     ,'DISTILL' : None
    #     ,'BASE':None
    #     }
    # elif model in ['hilbert_local_shift_rot_inv_dec_dq_opt_1633'] :
        
    #     MODEL_CONFIG = {
    #     'LOGNAME': model ,
    #     'find_unused_parameters':False,
    #     'MODEL_TYPE': (MotionMamba_hilbert_local_shift_rot_inv, MultiScaleFlow_dec_opt),
    #     'MODEL_ARCH': init_model_config(
    #         F = 16,
    #         motion=False,
    #         refine=dec_dq_opt,
    #         W=[8,8],
    #         depth = [2,2,2,3,3])
    #     ,'DISTILL' : None
    #     ,'BASE':None
    #     }

    # elif model in ['hilbert_local_shift_rot_dec_dq_opt_6422'] :
        
    #     MODEL_CONFIG = {
    #     'LOGNAME': model ,
    #     'find_unused_parameters':False,
    #     'MODEL_TYPE': (MotionMamba_hilbert_local_shift_rot, MultiScaleFlow_dec_opt),
    #     'MODEL_ARCH': init_model_config(
    #         F = 64,
    #         motion=False,
    #         refine=dec_dq_opt,
    #         W=[8,8],
    #         depth = [2,2,2,2,2])
    #     ,'DISTILL' : None
    #     ,'BASE':None
    #     }



#############3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d##########
#############3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d##########
#############3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d##########
#############3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d##########
#############3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d##########
#############3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d##########
#############3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d#######################3d##########




    elif model in ['hilbert_3d_local_shift_dec_dq_opt_1633'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_3d_local_shift, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,3,3])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['hilbert_3d_local_shift_rot_dec_dq_opt_1633'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_3d_local_shift_rot, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,3,3])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['hilbert_3d_local_shift_rot_dec_dq_opt_1644'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_3d_local_shift_rot, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
    elif model in ['hilbert_3d_local_shift_rot_dec_dq_opt_1644_w4'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_3d_local_shift_rot, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[4,4],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['hilbert_3d_local_shift_rot_dec_dq_opt_6444_w4'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_hilbert_3d_local_shift_rot, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 64,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }

    elif model in ['vfimamba'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor_mamba ,MultiScaleFlow_mamba),
        'MODEL_ARCH': init_model_config_mamba(
            F = 16,
            depth = [2, 2, 2, 3, 3],
            M = False,
        )
        ,'DISTILL' : None
        ,'BASE':None
    }

    elif model in ['vfimamba_big'] :
        
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (feature_extractor_mamba ,MultiScaleFlow_mamba),
        'MODEL_ARCH': init_model_config_mamba(
            F = 32,
            depth = [2, 2, 2, 4, 4],
            M = False,
        )
        ,'DISTILL' : None
        ,'BASE':None
    }
        

    elif model in ['localmamba'] :
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_localmamba, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
    
        

    elif model in ['continuousmamba'] :
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_continuousmamba, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
    

    elif model in ['crossmamba'] :
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_cross, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }


    elif model in ['bidirection'] :
        MODEL_CONFIG = {
        'LOGNAME': model ,
        'find_unused_parameters':False,
        'MODEL_TYPE': (MotionMamba_bidirection, MultiScaleFlow_dec_opt),
        'MODEL_ARCH': init_model_config(
            F = 16,
            motion=False,
            refine=dec_dq_opt,
            W=[8,8],
            depth = [2,2,2,4,4])
        ,'DISTILL' : None
        ,'BASE':None
        }
    
    return MODEL_CONFIG