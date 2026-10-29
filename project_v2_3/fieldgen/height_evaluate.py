"""Full-resolution high-building leave-one-out evaluation, after method selection."""
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
from .evaluate import stats,combine
from .labels import load_label


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--experiment',required=True)
    ap.add_argument('--method',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    meta=read(Path(a.dataset)/'manifest.json');exp=Path(a.experiment);plan=read(exp/'plan.json');out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    src=Path(meta['source_cache']);rows=[];per=[];snapshots=[]
    for case in plan['cases']:
        name=case['held_out'];ckpath=exp/'folds'/a.method/name/'best.pt';model,ck=load(ckpath,device)
        assert name not in ck['trained_buildings'];assert ck['dataset_manifest_sha256']==digest(Path(a.dataset)/'manifest.json')
        records=[r for r in meta['records'] if r['building']==name];ss=[];acc=[];slices=[];snapshot=None
        for r in records:
            xy,y,valid=load_label(meta,r,a.dataset)
            dims=r['dims'];w,l,h=dims
            roof=(abs(xy[...,0])<=l/2)&(abs(xy[...,1])<=w/2)&(abs(r['z']-h)<1e-6);valid&=~roof
            pred=infer(model,xy,dims,r['z'],device);s=stats(y[valid],pred[valid]);acc.append(s);score=combine([s])
            centres=binary_erosion(valid,np.ones((11,11),bool),border_value=0);sim=None
            if centres.any():
                _,sm=structural_similarity(np.where(valid,y,0),np.where(valid,pred,0),data_range=13,
                    gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
                sim=float(sm[centres].mean());ss.append(sim)
            row=dict(building=name,z=r['z'],**score,ssim=sim);rows.append(row);slices.append(row)
            distance=abs(r['ratio']-1.1)
            if snapshot is None or distance<snapshot[0]:snapshot=(distance,r,y,pred,valid,ck['palettes'])
        per.append(dict(building=name,in_training_dimension_bounds=case['in_training_dimension_bounds'],**combine(acc),
            mean_slice_r2=float(np.mean([r['r2'] for r in slices if r['r2'] is not None])),ssim=float(np.mean(ss)),slices=len(records),checkpoint_sha256=digest(ckpath)))
        snapshots.append(snapshot);print(name,'full RMSE',per[-1]['rmse'],flush=True)
    inside=[r for r in per if r['in_training_dimension_bounds']]
    write(out/'metrics.json',dict(method=a.method,per_building=per,
        in_range_macro_rmse=float(np.mean([r['rmse'] for r in inside])),all_five_macro_rmse=float(np.mean([r['rmse'] for r in per])),
        dataset_manifest_sha256=digest(Path(a.dataset)/'manifest.json'),
        scope='Each high building excluded from its model pretraining/adaptation. Four in-range cases, one L72 geometry extrapolation. Development evidence, not a fresh blind test. Final accepted Fluent magnitude labels in m/s; some cases do not meet strict residual criteria.'))
    write(out/'per_slice.json',rows)
    fig,axes=plt.subplots(len(per),3,figsize=(11,3.6*len(per)),layout='constrained')
    for i,(row,snapshot) in enumerate(zip(per,snapshots)):
        _,r,y,pred,valid,palettes=snapshot
        w,l,h=r['dims'];xy,_,_=load_label(meta,r,a.dataset)
        roof=(abs(xy[...,0])<=l/2)&(abs(xy[...,1])<=w/2)&(abs(r['z']-h)<1e-6)
        for j,v in enumerate((y,pred)):
            axes[i,j].imshow(np.dstack((render(np.where(roof,0,v),r['dims'][2],palettes),(valid|roof).astype('uint8')*255)))
        im=axes[i,2].imshow(np.where(valid,abs(pred-y),np.nan),cmap='magma',vmin=0,vmax=2)
        fig.colorbar(im,ax=axes[i,2],fraction=.04,label='m/s')
        for j,title in enumerate(('Fluent numerical magnitude','Held-out high building','Absolute speed error')):
            axes[i,j].set_title(title);axes[i,j].set_xticks([]);axes[i,j].set_yticks([]);axes[i,j].set_facecolor('#eeeeee')
        axes[i,0].set_ylabel(row['building']+f'\nz={r["z"]:g}m',fontsize=9)
    fig.savefig(out/'high_buildings.png',dpi=120);plt.close(fig)


if __name__=='__main__':main()
