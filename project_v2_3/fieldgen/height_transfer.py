"""Low-to-high building transfer with fixed budgets and whole-high-building exclusion."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from .common import ROOT,read,write,digest
from .train import Dataset,load
from .generalize import Sampler,validate
from .model import Field

DEFAULT_DATA=ROOT/'artifacts/dataset_verified'
METHODS=('joint_uniform','joint_balanced','transfer_replay','transfer_flexible')


def plan_for(meta):
    dims=dict(zip(meta['names'],meta['dims']))
    low=sorted(n for n,d in dims.items() if d[2]<=50)
    high=sorted(set(dims)-set(low));all_dims=np.array(meta['dims'])
    cases=[]
    for name in high:
        remaining=np.array([d for n,d in dims.items() if n!=name])
        inside=bool(np.all(np.array(dims[name])>=remaining.min(0)) and np.all(np.array(dims[name])<=remaining.max(0)))
        cases.append(dict(held_out=name,train=sorted(set(dims)-{name}),in_training_dimension_bounds=inside))
    return dict(low=low,high=high,cases=cases,seed=20260918,source_steps=2500,joint_steps=4000,
        adaptation_steps=1500,flexible_steps=1500,local_steps=300,adapter_warmup_steps=300,replay_low_fraction=.5,l2_sp=.1,
        sample_validation_points_per_slice=512,methods=list(METHODS),
        selection='Fixed endpoints, no held-out early stopping. Method selection uses equal-building weighted RMSE on the four in-range high-building leave-one-out cases. The L72 case is reported separately as dimension extrapolation.',
        caveat='Exploratory development comparison on existing data, not a new blind test. Same total 4300 training steps per trained method, including 2500 shared source steps for transfer.')


def mixed_indices(low,high,count=4096,low_fraction=.5):
    # Existing valid samples only: half source-domain replay, half high domain.
    nlow=int(count*low_fraction);a,b=low.indices(nlow),high.indices(count-nlow)
    return np.concatenate((a[0],b[0])),np.concatenate((a[1],b[1]))


def fit(dataset,out,plan,method,held_out=None,source=None,production=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError(out/'best.pt')
    ds=Dataset(dataset);torch.set_num_threads(4);torch.manual_seed(plan['seed'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.backends.cuda.matmul.allow_tf32=True
    names=plan['low'] if method=='source' else sorted(set(ds.meta['names'])-({held_out} if held_out else set()))
    assert held_out not in names
    architecture=dict(width=96,depth=4,output_mode='reference')
    if method in ('transfer_replay','transfer_flexible'):
        model,pre=load(source,device)
        assert set(pre['trained_buildings'])==set(plan['low']) and held_out not in pre['trained_buildings']
        origin={n:p.detach().clone() for n,p in model.named_parameters() if not n.startswith('local.')}
    else:model=Field(**architecture).to(device);origin={}
    uniform=Sampler(ds,names,True,plan['seed'])
    low=Sampler(ds,plan['low'],True,plan['seed']+1)
    high_names=sorted(set(names)&set(plan['high']))
    high=Sampler(ds,high_names,True,plan['seed']+2) if high_names else None
    steps=(plan['source_steps'] if method=='source' else plan['flexible_steps'] if method=='transfer_flexible'
           else plan['adaptation_steps'] if method=='transfer_replay' else plan['joint_steps'])
    local=0 if method=='source' else plan['local_steps'];history=[];started=time.time()
    for stage,total in [('global',steps),('local',local)]:
        if not total:continue
        for n,p in model.named_parameters():p.requires_grad_(n.startswith('local.') if stage=='local' else not n.startswith('local.'))
        learning_rate=5e-5 if stage=='local' else 2e-4 if method=='transfer_flexible' else 1e-4 if method=='transfer_replay' else .0007
        # Includes temporarily frozen global parameters, so warmup can unfreeze them later.
        optimizer=torch.optim.AdamW([p for n,p in model.named_parameters() if n.startswith('local.')==(stage=='local')],lr=learning_rate,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total,eta_min=learning_rate*.1)
        for step in range(1,total+1):
            if method in ('transfer_replay','transfer_flexible') and stage=='global':
                warm=step<=(150 if method=='transfer_flexible' else plan['adapter_warmup_steps'])
                for n,p in model.named_parameters():
                    p.requires_grad_(not n.startswith('local.') and (not warm or n.startswith(('condition.','head.'))))
            if method in ('joint_balanced','transfer_replay','transfer_flexible'):
                ii,jj=mixed_indices(low,high,low_fraction=.25 if method=='transfer_flexible' else .5)
            else:ii,jj=uniform.indices(4096)
            xy,d,r,y=ds.batch(ii,jj,device);model.train();pred,w=model(xy,d,r)
            ref=3*(d[:,2]/10).pow(.4)/13;loss=(((pred-y)/ref).square()*w).sum()/w.sum()
            if origin and stage=='global':
                penalty=sum((p-origin[n]).square().mean() for n,p in model.named_parameters() if n in origin and p.requires_grad)
                loss=loss+(0.01 if method=='transfer_flexible' else plan['l2_sp'])*penalty
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step()
            if step%500==0 or step==total:
                history.append(dict(stage=stage,step=step,train_loss=loss.item(),seconds=time.time()-started))
                write(out/'history.json',history);print(method,held_out,stage,step,round(loss.item(),6),flush=True)
    ck=dict(state=model.state_dict(),architecture=architecture,method=method,seed=plan['seed'],stage='fixed_endpoint_transfer' if source else 'fixed_endpoint_joint',
        production=production,trained_buildings=names,bounds=ds.meta['bounds'],inlet=ds.meta['config']['inlet'],palettes=ds.meta['palettes'],
        dataset_manifest_sha256=digest(ds.path/'manifest.json'),label_domains=ds.meta.get('label_domains'),
        training_steps=dict(source=plan['source_steps'] if source else 0,global_steps=steps,local_steps=local),
        source_checkpoint_sha256=digest(source) if source else None,held_out=held_out,
        training_support=[dict(name=n,dims=ds.meta['dims'][ds.meta['names'].index(n)],ratio_max=max(r['ratio'] for r in ds.rows if r['building']==n)) for n in names])
    torch.save(ck,out/'best.pt')
    score=validate(model,ds,[held_out],device,out/'held_out.npz') if held_out else None
    write(out/'run.json',dict(method=method,held_out=held_out,seconds=time.time()-started,validation=score,
                             checkpoint_sha256=digest(out/'best.pt'),held_out_used_for_training_or_early_stopping=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default=str(DEFAULT_DATA));ap.add_argument('--output',required=True)
    ap.add_argument('--plan',required=True);ap.add_argument('--method',choices=['source',*METHODS],required=True)
    ap.add_argument('--held-out');ap.add_argument('--source');ap.add_argument('--production',action='store_true')
    a=ap.parse_args();fit(a.dataset,a.output,read(a.plan),a.method,a.held_out,a.source,a.production)


if __name__=='__main__':main()
