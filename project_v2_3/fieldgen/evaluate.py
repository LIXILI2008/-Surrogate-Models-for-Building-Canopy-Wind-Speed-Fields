"""Whole-building test only AFTER model/continuity selection; no test-based tuning."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import read,write,digest
from .train import load
from .predict import infer,render
from .labels import load_label


def stats(y,p):
    y=y.astype(np.float64);p=p.astype(np.float64);e=p-y
    return dict(n=len(y),sy=float(y.sum()),sy2=float((y*y).sum()),se=float(e.sum()),se2=float((e*e).sum()),sa=float(abs(e).sum()),
        intersection=int(((y>=4)&(p>=4)).sum()),union=int(((y>=4)|(p>=4)).sum()))


def combine(rows):
    c={k:sum(r[k] for r in rows) for k in rows[0]};n=c['n'];sst=max(0.,c['sy2']-c['sy']**2/n)
    return dict(n=n,rmse=(c['se2']/n)**.5,mae=c['sa']/n,mean_bias=c['se']/n,
        r2=1-c['se2']/sst if sst>1e-8 else None,iou_at_4=c['intersection']/c['union'] if c['union'] else None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True);ap.add_argument('--dataset',default='artifacts/dataset');a=ap.parse_args()
    torch.set_num_threads(4);meta=read(Path(a.dataset)/'manifest.json');out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model,ck=load(Path(a.run)/'best.pt',device)
    assert not ck['production'];assert not set(ck['trained_buildings']) & set(meta['splits']['test'])
    assert ck['dataset_manifest_sha256']==digest(Path(a.dataset)/'manifest.json')
    groups={n:[] for n in meta['splits']['test']};bgroups={n:[] for n in groups};images=[];snapshots={}
    records=[r for r in meta['records'] if r['split']=='test'];src=Path(meta['source_cache'])
    for i,r in enumerate(records):
        xy,y,valid=load_label(meta,r,a.dataset)
        name=r['building'];dims=r['dims'];w,l,h=dims
        roof=(abs(xy[...,0])<=l/2)&(abs(xy[...,1])<=w/2)&(abs(r['z']-h)<1e-6);valid&=~roof
        pred=infer(model,xy,dims,r['z'],device);p=pred[valid];truth=y[valid]
        s=stats(truth,p);groups[name].append(s)
        bg=np.full_like(truth,3*(r['z']/10)**.4);bgroups[name].append(stats(truth,bg))
        centres=binary_erosion(valid,np.ones((11,11),bool),border_value=0)
        ss=None
        if centres.any():
            _,sm=structural_similarity(np.where(valid,y,0),np.where(valid,pred,0),data_range=6 if h<=50 else 13,
                gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
            ss=float(sm[centres].mean())
        images.append(dict(building=name,z_m=r['z'],ratio=r['ratio'],**combine([s]),ssim=ss,source_cache_hash_verified=True))
        if name not in snapshots or abs(r['ratio']-1.10)<snapshots[name][0]:snapshots[name]=(abs(r['ratio']-1.10),r,xy,y,pred,valid)
        if (i+1)%25==0:print('Test heights evaluated',i+1,flush=True)
    per=[]
    for n in groups:
        rr=[r for r in images if r['building']==n];ss=[r['ssim'] for r in rr if r['ssim'] is not None]
        per.append(dict(building=n,**combine(groups[n]),background=combine(bgroups[n]),
            mean_slice_r2=float(np.mean([r['r2'] for r in rr if r['r2'] is not None])),ssim=float(np.mean(ss)),slices=len(rr)))
    result=dict(model=str(Path(a.run)/'best.pt'),checkpoint_sha256=digest(Path(a.run)/'best.pt'),whole_building_holdout=True,
        trained_buildings=ck['trained_buildings'],test_buildings=meta['splits']['test'],per_building=per,
        pooled=combine([r for rows in groups.values() for r in rows]),macro_rmse=float(np.mean([r['rmse'] for r in per])),
        macro_background_rmse=float(np.mean([r['background']['rmse'] for r in per])),
        macro_ssim=float(np.mean([r['ssim'] for r in per])),
        evaluation_role='Previously viewed V2 test buildings, now regression reserve. Not a fresh blind test.',
        note='All heights of these cuboids excluded from this run training and selection. Their previous results were already viewed; this is regression evaluation. Scored only at existing valid source pixels; roof prescribed boundary excluded. Cache targets inherit V1 estimated registration.',
        no_claim='No empirical claim for all in-bounds combinations or source-unsupported heights; 3 test buildings only.')
    write(out/'metrics.json',result);write(out/'per_slice.json',images)
    fig,axes=plt.subplots(len(per),3,figsize=(11,3.8*len(per)),layout='constrained',squeeze=False)
    for i,n in enumerate(groups):
        _,r,xy,y,pred,valid=snapshots[n]
        truth_rgba=np.dstack((render(y,r['dims'][2],ck['palettes']),valid.astype('uint8')*255))
        pred_rgba=np.dstack((render(pred,r['dims'][2],ck['palettes']),valid.astype('uint8')*255))
        axes[i,0].imshow(truth_rgba);axes[i,1].imshow(pred_rgba)
        im=axes[i,2].imshow(np.where(valid,abs(pred-y),np.nan),cmap='magma',vmin=0,vmax=2);fig.colorbar(im,ax=axes[i,2],label='m/s',fraction=.04)
        for j,title in enumerate(('Registered source label','UNSEEN building prediction','Absolute speed error')):
            axes[i,j].set_title(title);axes[i,j].set_xticks([]);axes[i,j].set_yticks([]);axes[i,j].set_facecolor('#eeeeee')
        axes[i,0].set_ylabel(f"{n}\nz={r['z']:g} m",fontsize=9)
    fig.savefig(out/'unseen_buildings.png',dpi=135);plt.close(fig);print(result,flush=True)


if __name__=='__main__':main()
