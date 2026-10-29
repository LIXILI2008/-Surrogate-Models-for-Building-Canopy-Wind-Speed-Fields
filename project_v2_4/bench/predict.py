"""Specified geometry and height, using an explicitly chosen comparison checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write,digest,inlet
from .geometry import features,grid,masks_numpy
from .models import make,CoordinateModel
from .palette import render


def export(checkpoint,dims,z,output):
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    dims=np.asarray(dims,dtype=np.float32)
    if not np.isfinite(dims).all() or not np.isfinite(z):raise ValueError('Finite dimensions/height required')
    if np.any(dims<ck['bounds']['min']) or np.any(dims>ck['bounds']['max']):raise ValueError('Outside dataset dimension bounds')
    if not dims[2]<=z<=1.5*dims[2]:raise ValueError('Expected H<=z<=1.5H')
    size=128;w,l,h=dims;span=np.array([max(72.35,l+12),max(72.35,w+12)],np.float32)
    bounds=[-span[0]/2+span[0]/(2*size),-span[1]/2+span[1]/(2*size),
             span[0]/2-span[0]/(2*size),span[1]/2-span[1]/(2*size)]
    q=grid(bounds,dims,z,size);qt=torch.tensor(q,device=device)
    dt=torch.tensor(dims,device=device).expand(size,size,3)
    st=torch.tensor([bounds[2]-bounds[0],bounds[3]-bounds[1]],device=device,dtype=torch.float32).expand(size,size,2)
    f=features(qt,dt,st);model=make(ck['architecture']).to(device);model.load_state_dict(ck['state']);model.eval()
    with torch.no_grad():
        if isinstance(model,CoordinateModel):
            a=f.reshape(-1,17);values=torch.cat([model.from_features(v) for v in a.split(8192)]).reshape(size,size,7)
        else:values=model(f.permute(2,0,1)[None])[0].permute(1,2,0)
    values=values.cpu().numpy();speed=(values[...,0]+1)*inlet(z)
    projection,roof,fluid=masks_numpy(q,dims);speed[roof]=0
    output=Path(output);output.mkdir(parents=True,exist_ok=True);stem=f'W{w:g}_L{l:g}_H{h:g}_z{z:g}'
    np.savez_compressed(output/(stem+'.npz'),x=q[...,0]*h,y=q[...,1]*h,speed_mps=speed,
        relative_disturbance=speed/inlet(z)-1,inlet_mps=inlet(z),sdf_m=f[...,7].cpu().numpy()*h,
        sdf3_m=f[...,8].cpu().numpy()*h,projection=projection,roof=roof,fluid_mask=fluid,
        auxiliary_dimensionless=values[...,1:])
    # Consistent speed legend across models. Native 128 grid, not a claimed 512 prediction.
    plt.imsave(output/(stem+'.png'),render(speed,h))
    write(output/(stem+'.json'),dict(W=float(w),L=float(l),H=float(h),z=float(z),size=size,
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),architecture=ck['architecture'],
        checkpoint_held_out=ck['held_out'],production_model=False,color_range_mps=[0,6 if h<=50 else 13],
        color_map='Preserved V2.3 calibrated LUT, verified by SHA256; numerical labels remain SI',
        warning='Development fold checkpoint, not a selected production model. In-range combinations may still have weak support. Low-building auxiliary vectors/pressure/k/epsilon are not validated.'))
    return output/(stem+'.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    for name in ('W','L','H'):p.add_argument('--'+name,type=float,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--z',type=float);g.add_argument('--all-heights',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'outputs');a=p.parse_args()
    heights=[a.z] if a.z is not None else np.arange(a.H,1.5*a.H+1e-8,1.)
    for z in heights:print(export(a.checkpoint,[a.W,a.L,a.H],float(z),a.output))
