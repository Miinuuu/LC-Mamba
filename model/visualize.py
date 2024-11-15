import torch

def flow2rgb_tensor(flow_map):
    
    b ,c,h, w = flow_map.shape
    
    rgb_map = torch.ones((b,3,h, w),device=flow_map.device)

    max,_= torch.max(torch.abs(flow_map.view(b,-1)),dim=1)

    normalized_flow_map = flow_map / max.view(b,1, 1, 1)
    
    rgb_map[:,0,:, :] += normalized_flow_map[:, 0]
    rgb_map[:,1,:, :] -= 0.5 * (normalized_flow_map[:, 0] + normalized_flow_map[:, 1])
    rgb_map[:,2,:, :] += normalized_flow_map[:, 1]
    return rgb_map.clip(0, 1)

