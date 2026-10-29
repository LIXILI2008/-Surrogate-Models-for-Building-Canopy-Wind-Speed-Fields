import numpy as np
import torch

CHANNELS = ['x/H', 'y/H', 'z/H', 'W/H', 'L/H', 'log(H/20)',
            'projection', 'sdf/H', 'sdf3/H', 'fluid', 'roof', 'Uin/UH',
            'log(W/16)', 'log(L/24)', 'local_gate', 'span_x/H', 'span_y/H']


def features(q, dims, span):
    """q=(x/H,y/H,z/H). Continuous inputs; masks are analytic, never label-derived."""
    w, l, h = dims.unbind(-1); x, y, r = q.unbind(-1)
    # Smooth the absolute/norm singularities for PINN second derivatives.
    ax = torch.sqrt(x*x+1e-12); ay = torch.sqrt(y*y+1e-12)
    qx = ax-l/(2*h); qy = ay-w/(2*h)
    sdf = torch.sqrt(qx.clamp_min(0)**2+qy.clamp_min(0)**2+1e-12) + torch.maximum(qx,qy).clamp_max(0)
    qz = torch.sqrt((r-.5)**2+1e-12)-.5
    sdf3 = torch.sqrt(qx.clamp_min(0)**2+qy.clamp_min(0)**2+qz.clamp_min(0)**2+1e-12) + torch.maximum(torch.maximum(qx,qy),qz).clamp_max(0)
    projection = (x.abs()<=l/(2*h)) & (y.abs()<=w/(2*h))
    roof = projection & ((r-1).abs()<1e-6)
    solid = projection & (r<1-1e-6)
    boundary = torch.exp(-sdf.abs()*h/1.5)
    nearroof = torch.exp(-(r-1).abs()/.08)*torch.exp(-sdf.clamp_min(0)*h/3)
    gate = torch.maximum(boundary,nearroof)
    return torch.stack((x,y,r,w/h,l/h,torch.log(h/20),projection.float(),sdf,sdf3,
                        (~solid & ~roof).float(),roof.float(),r.clamp_min(.01)**.4,
                        torch.log(w/16),torch.log(l/24),gate,span[...,0]/h,span[...,1]/h),-1)


def grid(bounds, dims, z, size):
    x0,y0,x1,y1 = bounds
    xx,yy = np.meshgrid(np.linspace(x0,x1,size),np.linspace(y0,y1,size))
    return np.stack((xx/dims[2],yy/dims[2],np.full_like(xx,z/dims[2])),-1).astype('float32')


def masks_numpy(q,dims):
    w,l,h = dims; projection=(abs(q[...,0])<=l/(2*h)) & (abs(q[...,1])<=w/(2*h))
    roof=projection & (abs(q[...,2]-1)<1e-6)
    solid=projection & (q[...,2]<1-1e-6)
    return projection,roof,~(roof|solid)
