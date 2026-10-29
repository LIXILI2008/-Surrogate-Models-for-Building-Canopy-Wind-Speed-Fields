"""One specified W,L,H,z -> one PNG plus numerical field and provenance."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from scipy.spatial import Delaunay,QhullError
from .common import ROOT,check_query,digest,write
from .geometry import query_grid,features
from .train import load


def render(speed,h,palettes):
    filename='real_lut_0_6.csv' if h<=50 else 'real_lut_0_13.csv'
    expected=next(p for p in palettes if p['filename']==filename)
    path=ROOT/'configs'/filename
    if digest(path)!=expected['sha256']:raise ValueError('Calibration hash mismatch')
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    values=np.array([float(r['value']) for r in rows]);colors=np.array([[float(r[c]) for c in ('r','g','b')] for r in rows])
    order=np.argsort(values);values=values[order];colors=colors[order]
    return np.stack([np.interp(speed,values,colors[:,c]) for c in range(3)],-1).round().clip(0,255).astype('uint8')


@torch.no_grad()
def infer(model,xy,dims,z,device,return_spread=False):
    shape=xy.shape[:-1];coords=xy.reshape(-1,2);out=[];spread=[]
    for start in range(0,len(coords),16384):
        points=torch.as_tensor(coords[start:start+16384],device=device)
        d=torch.tensor(dims,dtype=torch.float32,device=device).expand(len(points),3)
        r=torch.full((len(points),),float(z/dims[2]),device=device)
        if return_spread and hasattr(model,'members'):
            values=torch.stack([m(points,d,r)[0] for m in model.members])
            out.append(values.mean(0).cpu().numpy()*13)
            spread.append(values.std(0,unbiased=False).cpu().numpy()*13)
        else:
            out.append(model(points,d,r)[0].cpu().numpy()*13)
            if return_spread:spread.append(np.zeros(len(points),np.float32))
    mean=np.concatenate(out).reshape(shape)
    return (mean,np.concatenate(spread).reshape(shape)) if return_spread else mean


def support_info(ck,dims,z):
    dims=np.asarray(dims,float);known=np.array([r['dims'] for r in ck['training_support']]);span=np.array(ck['bounds']['max'])-ck['bounds']['min']
    dist=np.linalg.norm((known-dims)/span,axis=1);ids=np.argsort(dist)[:4]
    try:inside=bool(Delaunay(known/span).find_simplex(dims/span)>=0)
    except QhullError:inside=False
    neighbours=[dict(**ck['training_support'][int(i)],normalized_distance=float(dist[i])) for i in ids]
    return dict(known_geometry=bool((dist<1e-7).any()),inside_training_geometry_convex_hull=inside,
        nearest_training_geometries=neighbours,
        height_exceeds_any_nearby_support=bool(any(z/dims[2]>r['ratio_max']+1e-6 for r in neighbours)),
        note='In-bounds does not guarantee dense training coverage. Neighbour distance/hull are support diagnostics, not calibrated uncertainty.')


def export(checkpoint,dims,z,output,size=512):
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,ck=load(checkpoint,device);dims=check_query(dims,z,ck['bounds']);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    xy,span=query_grid(dims,size);speed,spread=infer(model,xy,dims,z,device,return_spread=True)
    w,l,h=dims;footprint=(np.abs(xy[...,0])<=l/2)&(np.abs(xy[...,1])<=w/2)
    roof=footprint & (abs(z-h)<1e-6);domain=np.ones(speed.shape,bool);fluid=domain & ~roof
    tensorxy=torch.from_numpy(xy.reshape(-1,2));d=torch.tensor(dims,dtype=torch.float32).expand(len(tensorxy),3)
    f,*_=features(tensorxy,d,torch.full((len(tensorxy),),float(z/h)))
    sdf=(f[:,4]*h).reshape(speed.shape).numpy();sdf3=(f[:,5]*h).reshape(speed.shape).numpy()
    bg=float(3*(z/10)**.4)
    stem=f'W{w:.12g}_L{l:.12g}_H{h:.12g}_z{z:.12g}m'
    meta=dict(W_m=w,L_m=l,H_m=h,z_m=float(z),z_over_H=float(z/h),units='m/s',inlet_mps=bg,
        checkpoint_sha256=digest(checkpoint),model_stage=ck['stage'],production_model=ck['production'],
        ensemble_members=len(ck.get('members',[ck])),ensemble_spread_note='Population standard deviation of member speeds; not calibrated error or confidence. Zero for a single model.',
        field_of_view_m=span.tolist(),pixels=size,geometry_convention='L is along wind x; W across wind y; cuboid centered at (0,0)',
        height_method='Direct arbitrary real z query of the trained continuous field; no rounding and no post-hoc average of neighbouring predictions.',
        roof_policy='At exact H, stationary no-slip roof=0; roof boundary rendered blue, separate from fluid mask.',
        speed_source='Scalar surrogate: low calibrated image labels and high accepted Fluent numerical magnitude labels. Prediction is not a CFD convergence guarantee.',
        support=support_info(ck,dims,z),display_range_mps=[0,6 if h<=50 else 13],
        display_saturated_fraction=float((speed>(6 if h<=50 else 13)).mean()))
    rgba=np.dstack((render(speed,h,ck['palettes']),np.full(speed.shape,255,np.uint8)))
    info=PngInfo();info.add_text('z_m',str(z));info.add_text('units','m/s');info.add_text('scope','unseen geometry estimate')
    Image.fromarray(rgba).save(out/(stem+'.png'),pnginfo=info)
    np.savez_compressed(out/(stem+'.npz'),speed_mps=speed,inlet_mps=np.float32(bg),disturbance_mps=speed-bg,
        x_m=xy[...,0],y_m=xy[...,1],sdf_m=sdf,sdf3_m=sdf3,footprint=footprint,valid_fluid=fluid,roof_boundary=roof,query_domain=domain,z_m=float(z),ensemble_spread_mps=spread)
    write(out/(stem+'.json'),meta)
    print('ONE requested-height PNG:',out/(stem+'.png'),flush=True)
    return meta


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='artifacts/height_transfer_numeric_verified/production/best.pt')
    ap.add_argument('--W',type=float,required=True,help='across-wind width, metres');ap.add_argument('--L',type=float,required=True,help='along-wind length, metres')
    ap.add_argument('--H',type=float,required=True);ap.add_argument('--z',type=float,required=True);ap.add_argument('--output',default='outputs');ap.add_argument('--size',type=int,default=512)
    a=ap.parse_args();export(a.checkpoint,[a.W,a.L,a.H],a.z,a.output,a.size)


if __name__=='__main__':main()
