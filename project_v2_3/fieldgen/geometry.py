"""Analytic geometry: no building ID, image, or dataset mask needed at inference."""
import numpy as np
import torch


def features(xy,dims,ratio):
    """Tensor inputs in metres; ratio=z/H. Return features, conditions, background, gate, roof, weights."""
    w,l,h=dims.unbind(-1);x,y=xy.unbind(-1);z=ratio*h
    q=torch.stack((x.abs()-l/2,y.abs()-w/2),-1)
    sdf=torch.linalg.vector_norm(q.clamp_min(0),dim=-1)+q.max(-1).values.clamp_max(0)
    projection=(q.max(-1).values<=0)
    roof=projection & ((ratio-1).abs()<1e-7)
    solid=projection & (ratio<1-1e-7)
    v=(z-h/2).abs()-h/2
    sdf3=torch.sqrt(sdf.clamp_min(0)**2+v.clamp_min(0)**2)+torch.maximum(sdf,v).clamp_max(0)
    tx=x/(l/2);ty=y/(w/2)
    f=[tx/4,ty/4,x/h,y/h,sdf/h,sdf3/h,projection.float(),(~solid).float(),roof.float(),ratio-1,
       (ratio-1)*h/10,torch.log1p((ratio-1).clamp_min(0)*h),torch.exp(-(ratio-1).clamp_min(0)*20)]
    for value in (tx/4,ty/4):
        for freq in (1,2,4):f.extend((torch.sin(np.pi*freq*value),torch.cos(np.pi*freq*value)))
    spatial=torch.stack(f,-1)
    condition=torch.stack((torch.log(w/16),torch.log(l/24),torch.log(h/20),torch.log(w/h),torch.log(l/h)),dim=-1)
    background=3*(z/10).clamp_min(0).pow(.4)/13
    boundary=torch.exp(-sdf.abs()/1.5)
    nearroof=torch.exp(-(z-h).abs()/torch.maximum(h*.08,torch.ones_like(h)))*torch.exp(-sdf.clamp_min(0)/3)
    gate=torch.maximum(boundary,nearroof)
    return spatial,condition,background,gate,roof,1+4*boundary+6*nearroof


def query_grid(dims,size=512,fov=None):
    w,l,h=map(float,dims)
    span=np.array(fov if fov is not None else [max(512/7.077313,l+12),max(512/7.077313,w+12)])
    if np.any(span<=np.array([l,w])):raise ValueError('FOV must contain the complete footprint')
    x=(np.arange(size)+.5-size/2)*span[0]/size
    y=(np.arange(size)+.5-size/2)*span[1]/size
    xx,yy=np.meshgrid(x,y)
    return np.stack((xx,yy),-1).astype(np.float32),span
