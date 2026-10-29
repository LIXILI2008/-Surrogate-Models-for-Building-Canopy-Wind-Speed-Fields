"""Previously viewed low-building regression holdouts, excluded from each fit.
For transfer candidates, each held-out low building is also excluded from source pretraining.
These results never select the high-domain method.
"""
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
from .height_transfer import plan_for,fit
from .train import load
from .labels import load_label
from .predict import infer,render
from .evaluate import stats,combine

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--experiment',required=True);a=ap.parse_args()
    ds=Path(a.dataset);exp=Path(a.experiment);out=exp/'low_regression';out.mkdir(exist_ok=True)
    meta=read(ds/'manifest.json');selected=read(exp/'selection.json')['selected'];plan=plan_for(meta)
    names=[n for n in meta['splits']['test'] if n in plan['low']]
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4);per=[];snapshots=[]
    for name in names:
        target=out/name;target.mkdir(exist_ok=True);local_plan=dict(plan,low=[n for n in plan['low'] if n!=name]);source=None
        if selected.startswith('transfer_'):
            source=target/'source/best.pt'
            if not (target/'source/run.json').exists():fit(ds,target/'source',local_plan,'source')
            _,source_ck=load(source,'cpu');assert name not in source_ck['trained_buildings']
        if not (target/'fit/run.json').exists():fit(ds,target/'fit',local_plan,selected,held_out=name,source=source)
        model,ck=load(target/'fit/best.pt',device);assert name not in ck['trained_buildings']
        assert ck['dataset_manifest_sha256']==digest(ds/'manifest.json')
        rows=[];acc=[];snapshot=None
        for r in [r for r in meta['records'] if r['building']==name]:
            xy,y,valid=load_label(meta,r,ds);p=infer(model,xy,r['dims'],r['z'],device)
            s=stats(y[valid],p[valid]);acc.append(s);sim=None
            centres=binary_erosion(valid,np.ones((11,11),bool),border_value=0)
            if centres.any():
                _,sm=structural_similarity(np.where(valid,y,0),np.where(valid,p,0),data_range=6,
                    gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
                sim=float(sm[centres].mean())
            rows.append(dict(z=r['z'],**combine([s]),ssim=sim))
            delta=abs(r['ratio']-1.1)
            if snapshot is None or delta<snapshot[0]:snapshot=(delta,r,xy,y,p,valid,ck['palettes'])
        r2=[r['r2'] for r in rows if r['r2'] is not None];ss=[r['ssim'] for r in rows if r['ssim'] is not None]
        score=dict(building=name,**combine(acc),ssim=float(np.mean(ss)) if ss else None,
            mean_slice_r2=float(np.mean(r2)) if r2 else None,slices=len(rows),checkpoint_sha256=digest(target/'fit/best.pt'))
        write(target/'metrics.json',score);write(target/'per_slice.json',rows);per.append(score);snapshots.append(snapshot)
        print('LOW REGRESSION',name,score['rmse'],flush=True)
    fig,axes=plt.subplots(len(per),3,figsize=(11,3.8*len(per)),layout='constrained',squeeze=False)
    for i,(score,snap) in enumerate(zip(per,snapshots)):
        _,r,xy,y,p,valid,palettes=snap;w,l,h=r['dims']
        roof=(abs(xy[...,0])<=l/2)&(abs(xy[...,1])<=w/2)&(abs(r['z']-h)<1e-6)
        for j,v in enumerate((y,p)):axes[i,j].imshow(np.dstack((render(np.where(roof,0,v),h,palettes),(valid|roof).astype('uint8')*255)))
        im=axes[i,2].imshow(np.where(valid,abs(p-y),np.nan),cmap='magma',vmin=0,vmax=1)
        fig.colorbar(im,ax=axes[i,2],label='m/s',fraction=.04)
        for j,title in enumerate(('Calibrated low label','Held-out low building','Absolute error')):
            axes[i,j].set_title(title);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
        axes[i,0].set_ylabel(score['building'])
    fig.savefig(out/'low_buildings.png',dpi=130);plt.close(fig)
    write(out/'metrics.json',dict(method=selected,per_building=per,macro_rmse=float(np.mean([r['rmse'] for r in per])),
        scope='Two previously viewed low-building regression cases. Each excluded from both source pretraining and adaptation. Not used for method selection; not a fresh blind test.'))

if __name__=='__main__':main()
