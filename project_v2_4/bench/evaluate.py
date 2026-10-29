import argparse
import sys
import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,WORKSPACE,read,write,digest,inlet
from .data import Dataset
from .models import make,CoordinateModel
from .geometry import grid
from .metrics import score,scalar
from .physics_fd import continuity


def average(values):
    v=[float(x) for x in values if x is not None]
    return float(np.mean(v)) if v else None


def aggregate(rows):
    result={}
    for key in ('speed','disturbance','geometric_wake','roof_neighbourhood'):
        total=sum(r[key]['n'] for r in rows)
        result[key]=dict(n=total,rmse=(sum(r[key]['n']*r[key]['rmse']**2 for r in rows if r[key]['n'])/total)**.5 if total else None,
            mae=sum(r[key]['n']*r[key]['mae'] for r in rows if r[key]['n'])/total if total else None,
            mean_slice_r2=average([r[key]['r2'] for r in rows]))
    for key in ('iou_at_4','deficit_0.1','deficit_0.2','acceleration_0.1','acceleration_0.2','wake_deficit_0.1','wake_deficit_0.2'):
        totals={k:sum(r[key][k] for r in rows) for k in ('n','truth_positive','pred_positive','intersection','union')}
        t=totals;totals.update(iou=t['intersection']/t['union'] if t['union'] else None,
            truth_fraction=t['truth_positive']/t['n'] if t['n'] else None,
            false_positive_fraction=(t['pred_positive']-t['intersection'])/t['n'] if t['n'] else None,
            empty_truth_slices=sum(r[key]['truth_empty'] for r in rows),
            empty_union_slices=sum(r[key]['union']==0 for r in rows))
        result[key]=totals
    result['ssim_speed']=average([r['ssim_speed'] for r in rows])
    result['ssim_disturbance']=average([r['ssim_disturbance'] for r in rows])
    return result


def evaluate(dataset,run,held_out,legacy=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ds=Dataset(dataset);device='cuda' if torch.cuda.is_available() else 'cpu'
    if (run/'metrics.json').exists():return read(run/'metrics.json')
    if legacy:
        # Import the preserved V2.3 implementation under its own package name.
        sys.path.insert(0,str(WORKSPACE/'project_v2_3'))
        from fieldgen.train import load
        from fieldgen.predict import infer
        exp=WORKSPACE/'project_v2_3/artifacts/height_transfer_numeric_verified'
        checkpoint=(exp/'low_regression'/held_out/'fit/best.pt' if ds.dimensions[held_out][2]<=50 else
                    exp/'folds/joint_balanced'/held_out/'best.pt')
        model,ck=load(checkpoint,device)
    else:
        checkpoint=run/'best.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert ck['signature']['dataset_manifest_sha256']==digest(dataset/'manifest.json')
        model=make(ck['architecture']).to(device);model.load_state_dict(ck['state']);model.eval()
    assert held_out not in ck['trained_buildings']
    result=[];background=[];ys=[];ps=[];ds_y=[];ds_p=[];snapshot=None;vector=[]
    pred_volume=[];true_volume=[];volume_valid=[];ratios=[]
    for idx in ds.ids[held_out]:
        row=ds.rows[idx];batch=ds.batch([idx],device);dims=row['dims'];z=row['z'];q=batch['q'][0].cpu().numpy()
        true=ds.target[idx,0];valid=ds.valid[idx];aux=ds.aux_valid[idx]
        with torch.no_grad():
            if legacy:pred=infer(model,q[...,:2]*dims[2],dims,z,device)/inlet(z)-1;out=None
            elif isinstance(model,CoordinateModel):
                outputs=[];f=batch['f'][0].reshape(-1,17)
                for start in range(0,len(f),8192):outputs.append(model.from_features(f[start:start+8192]).cpu().numpy())
                out=np.concatenate(outputs).reshape(ds.meta['size'],ds.meta['size'],7);pred=out[...,0]
            else:
                out=model(batch['f'].permute(0,3,1,2))[0].permute(1,2,0).cpu().numpy();pred=out[...,0]
        if out is not None and aux.any():
            e=(out[...,1:4]-ds.target[idx,1:4].transpose(1,2,0))*inlet(dims[2])
            vector.append(dict(n=int(aux.sum()),sse=float((e[aux].astype('float64')**2).sum())))
            pred_volume.append(out[...,1:4]);true_volume.append(ds.target[idx,1:4].transpose(1,2,0))
            volume_valid.append(aux.copy());ratios.append(z/dims[2])
        s=score(true,pred,valid,q,dims,z);s['z']=z;result.append(s)
        background.append(score(true,np.zeros_like(pred),valid,q,dims,z))
        ys.append((true[valid]+1)*inlet(z));ps.append((pred[valid]+1)*inlet(z));ds_y.append(true[valid]);ds_p.append(pred[valid])
        distance=abs(z/dims[2]-1.1)
        if snapshot is None or distance<snapshot[0]:snapshot=(distance,true.copy(),pred.copy(),valid.copy(),z,dims)
    pooled=aggregate(result);pooled['speed'].update(scalar(np.concatenate(ys),np.concatenate(ps)))
    pooled['disturbance'].update(scalar(np.concatenate(ds_y),np.concatenate(ds_p)))
    pooled['aux_vector_component_rmse_mps']=(sum(v['sse'] for v in vector)/(3*sum(v['n'] for v in vector)))**.5 if vector else None
    if pred_volume:
        order=np.argsort(ratios);rr=np.array(ratios)[order];vv=np.stack(volume_valid)[order]
        pooled['continuity_predicted']=continuity(np.stack(pred_volume)[order],vv,rr,row['bounds'],dims[2])
        pooled['continuity_cfd_reference']=continuity(np.stack(true_volume)[order],vv,rr,row['bounds'],dims[2])
    pooled.update(building=held_out,slices=len(result),checkpoint_sha256=digest(checkpoint),resolution=ds.meta['size'],
                  inlet_only_baseline=aggregate(background))
    run.mkdir(parents=True,exist_ok=True);write(run/'per_slice.json',result);write(run/'metrics.json',pooled)
    _,true,pred,valid,z,dims=snapshot;u0=inlet(z)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    panels=[((true+1)*u0,'True speed','turbo',0,6 if dims[2]<=50 else 13),
            ((pred+1)*u0,'Predicted speed','turbo',0,6 if dims[2]<=50 else 13),
            (abs(pred-true)*u0,'Absolute error (m/s)','magma',0,1),
            (true,'True relative disturbance','coolwarm',-.5,.5),
            (pred,'Predicted relative disturbance','coolwarm',-.5,.5),
            ((pred<=-.1).astype(float)-(true<=-.1).astype(float),'Deficit errors: red FP, blue FN','bwr',-1,1)]
    for ax,(v,title,cmap,vmin,vmax) in zip(axes.flat,panels):
        im=ax.imshow(np.where(valid,v,np.nan),cmap=cmap,vmin=vmin,vmax=vmax);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle(f'{held_out}, z={z:g}m, {run.parent.name} / {run.name}')
    fig.savefig(run/'comparison.png',dpi=140);plt.close(fig)
    return pooled


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default=str(ROOT/'artifacts/dataset_128'))
    p.add_argument('--run',required=True);p.add_argument('--held-out',required=True);p.add_argument('--legacy',action='store_true');a=p.parse_args()
    evaluate(Path(a.dataset),Path(a.run),a.held_out,a.legacy)
