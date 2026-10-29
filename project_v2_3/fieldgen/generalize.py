"""V2.1: geometry-group CV, height/region sampling, reference-speed residuals.

No synthetic geometry labels, image flips, or new CFD data are introduced.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from .common import ROOT,read,write,digest
from .model import Field
from .train import Dataset,load

DEFAULT_DATA=ROOT.parent/'project_v2/artifacts/dataset'
VARIANTS={'baseline':dict(balanced=False,output_mode='legacy'),
          'balanced':dict(balanced=True,output_mode='legacy'),
          'balanced_reference':dict(balanced=True,output_mode='reference')}


def height_bin(ratio):
    return np.clip(np.floor((np.asarray(ratio)-1)*10+1e-6).astype(int),0,4)


def regions(xy,dims):
    """Disjoint geometry-based strata: footprint, boundary, wake, remainder."""
    w,l,h=np.asarray(dims);x,y=np.asarray(xy).T
    q=np.stack((abs(x)-l/2,abs(y)-w/2),-1)
    sdf=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
    result=np.full(len(x),3,np.int8)
    result[(x>l/2)&(abs(y)<w/2+.3*np.maximum(x-l/2,0))]=2
    result[abs(sdf)<1.5]=1
    result[(abs(x)<=l/2)&(abs(y)<=w/2)]=0
    return result


def make_plan(meta,folds=5):
    reserve=set(meta['splits']['test']);dev=sorted(set(meta['names'])-reserve)
    dims={n:np.asarray(meta['dims'][meta['names'].index(n)]) for n in dev}
    dd=np.stack(list(dims.values()));anchors=set()
    for axis in range(3):
        for extreme in (dd[:,axis].min(),dd[:,axis].max()):
            matching=[n for n in dev if dims[n][axis]==extreme]
            if len(matching)==1:anchors.update(matching)
    eligible=sorted(set(dev)-anchors,key=lambda n:(dims[n][2],dims[n][0],dims[n][1]))
    rng=np.random.default_rng(20260908)
    for attempt in range(1000):
        groups=[[] for _ in range(folds)]
        for start in range(0,len(eligible),folds):
            block=eligible[start:start+folds]
            for n,k in zip(block,rng.permutation(folds)):groups[k].append(n)
        valid=True
        for group in groups:
            train=np.stack([dims[n] for n in dev if n not in group])
            valid &= np.array_equal(train.min(0),dd.min(0)) and np.array_equal(train.max(0),dd.max(0))
        if valid:break
    else:raise ValueError('Cannot construct in-range group folds')
    return dict(seed=20260908,development=dev,anchors=sorted(anchors),regression_reserve=sorted(reserve),
        folds=[dict(train=sorted(set(dev)-set(g)),validation=sorted(g)) for g in groups],
        scope='CV on non-anchor development buildings; anchors retain dimension extrema in every training fold. Previously viewed test buildings reserved for regression, not fresh blind testing.',
        variants=VARIANTS,steps=4000,local_steps=300,eval_every=400,batch_points=4096,
        seeds=[20260908,20261009],criterion='Equal-building macro physical weighted RMSE on grouped out-of-fold validation; no regression-reserve selection.')


class Sampler:
    def __init__(self,ds,names,balanced,seed):
        self.ds=ds;self.names=list(names);self.balanced=balanced;self.rng=np.random.default_rng(seed)
        self.ids={n:np.array([i for i,r in enumerate(ds.rows) if r['building']==n]) for n in names}
        self.bins={n:[ix[height_bin(ds.ratio[ix])==b] for b in range(5) if np.any(height_bin(ds.ratio[ix])==b)] for n,ix in self.ids.items()}
        self.strata={}
        if balanced:
            for ids in self.ids.values():
                for i in ids:
                    labels=regions(ds.points[i,:,:2],ds.dims[i])
                    self.strata[i]=[np.flatnonzero(labels==k) for k in range(4) if np.any(labels==k)]

    def choose_slice(self,name):
        if self.balanced:
            options=self.bins[name]
            return int(self.rng.choice(options[self.rng.integers(len(options))]))
        return int(self.rng.choice(self.ids[name]))

    def indices(self,count):
        ii=[];jj=[]
        for name in self.rng.choice(self.names,size=4):
            i=self.choose_slice(name);n=count//4
            pp=self.rng.integers(self.ds.points.shape[1],size=n)
            if self.balanced:
                # Half uniform coverage; half uniform over nonempty geometric strata.
                strata=self.strata[i];which=self.rng.integers(len(strata),size=n//2)
                for k,points in enumerate(strata):
                    loc=np.flatnonzero(which==k);pp[loc]=self.rng.choice(points,size=len(loc))
            ii.extend([i]*n);jj.extend(pp)
        return np.array(ii),np.array(jj)


def summaries(ds,indices,y,p,w):
    rows=[]
    for name in sorted({ds.rows[int(i)]['building'] for i in indices}):
        keep=np.array([ds.rows[int(i)]['building']==name for i in indices]);e=(p[keep]-y[keep])*13
        truth=y[keep]*13;pred=p[keep]*13;ww=w[keep];sst=np.square(truth-truth.mean()).sum()
        union=((truth>=4)|(pred>=4)).sum()
        rows.append(dict(building=name,n=int(keep.sum()),rmse=float(np.sqrt(np.mean(e**2))),
            weighted_rmse=float(np.sqrt(np.sum(e**2*ww)/ww.sum())),mae=float(np.mean(abs(e))),bias=float(e.mean()),
            r2=float(1-np.square(e).sum()/sst) if sst>1e-9 else None,
            iou_at_4=float(((truth>=4)&(pred>=4)).sum()/union) if union else None))
    return dict(macro_rmse=float(np.mean([r['rmse'] for r in rows])),
                macro_weighted_rmse=float(np.mean([r['weighted_rmse'] for r in rows])),buildings=rows)


@torch.no_grad()
def validate(model,ds,names,device,save=None):
    ids=[i for i,r in enumerate(ds.rows) if r['building'] in names]
    ii=np.repeat(ids,512);jj=np.tile(np.arange(512),len(ids));ys=[];ps=[];ws=[]
    model.eval()
    for start in range(0,len(ii),16384):
        xy,d,r,y=ds.batch(ii[start:start+16384],jj[start:start+16384],device);p,w=model(xy,d,r)
        ys.append(y.cpu().numpy());ps.append(p.cpu().numpy());ws.append(w.cpu().numpy())
    y=np.concatenate(ys).astype(np.float64);p=np.concatenate(ps).astype(np.float64);w=np.concatenate(ws).astype(np.float64)
    result=summaries(ds,ii,y,p,w)
    if save:
        np.savez_compressed(save,indices=ii,target=y,pred=p,weight=w)
    return result


def fit(dataset,out,variant,seed,names,val_names,steps=4000,local_steps=300,production=False,factorized_rank=0):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError(out/'best.pt')
    torch.set_num_threads(4);torch.manual_seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.backends.cuda.matmul.allow_tf32=True
    ds=Dataset(dataset);config=VARIANTS[variant];sampler=Sampler(ds,names,config['balanced'],seed)
    arch=dict(width=96,depth=4,output_mode=config['output_mode'])
    if factorized_rank:arch['factorized_rank']=factorized_rank
    model=Field(**arch).to(device)
    reference=config['output_mode']=='reference';best=float('inf');global_step=steps;history=[];start=time.time()
    manifest_hash=digest(ds.path/'manifest.json')
    def save(stage,step,validation):
        torch.save(dict(state=model.state_dict(),architecture=arch,variant=variant,seed=seed,stage=stage,step=step,
            global_step=global_step,production=production,trained_buildings=list(names),bounds=ds.meta['bounds'],
            inlet=ds.meta['config']['inlet'],palettes=ds.meta['palettes'],validation=validation,
            dataset_manifest_sha256=manifest_hash,
            training_support=[dict(name=n,dims=ds.meta['dims'][ds.meta['names'].index(n)],
                                   ratio_max=max(ds.rows[i]['ratio'] for i in sampler.ids[n])) for n in names]),out/'best.pt')
    for stage,limit in [('global',steps),('local',local_steps)]:
        if stage=='local':
            model,prior=load(out/'best.pt',device);global_step=prior['global_step']
        for name,p in model.named_parameters():p.requires_grad_(name.startswith('local.') if stage=='local' else not name.startswith('local.'))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5 if stage=='local' else .0007,weight_decay=.0001)
        # Keep the same 4000-step learning-rate trajectory for all-data refits.
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,4000,eta_min=.00007) if stage=='global' else None
        for step in range(1,limit+1):
            model.train();ii,jj=sampler.indices(4096);xy,d,r,y=ds.batch(ii,jj,device)
            pred,w=model(xy,d,r);error=pred-y
            if reference:error=error/(3*(d[:,2]/10).pow(.4)/13)
            loss=(error.square()*w).sum()/w.sum()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if schedule:schedule.step()
            if step%400==0 or step==limit:
                validation=None if not val_names else validate(model,ds,val_names,device)
                history.append(dict(stage=stage,step=step,seconds=time.time()-start,loss=loss.item(),validation=validation))
                write(out/'history.json',history)
                if not val_names:
                    if step==limit:
                        if stage=='global':global_step=step
                        save(stage,step,None)
                elif validation['macro_weighted_rmse']<best:
                    best=validation['macro_weighted_rmse']
                    if stage=='global':global_step=step
                    save(stage,step,validation)
                print(out.name,stage,step,'weighted_rmse',None if validation is None else round(validation['macro_weighted_rmse'],5),flush=True)
    model,ck=load(out/'best.pt',device)
    if val_names:validation=validate(model,ds,val_names,device,out/'oof.npz')
    else:validation=None
    result=dict(variant=variant,factorized_rank=factorized_rank,seed=seed,seconds=time.time()-start,validation=validation,
                global_step=ck['global_step'],local_step=ck['step'] if ck['stage']=='local' else 0,checkpoint_sha256=digest(out/'best.pt'))
    write(out/'run.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default=str(DEFAULT_DATA));ap.add_argument('--output',required=True)
    ap.add_argument('--variant',choices=VARIANTS,default='baseline');ap.add_argument('--seed',type=int,default=20260908)
    ap.add_argument('--plan',required=True);ap.add_argument('--fold',type=int);ap.add_argument('--refit',choices=['development','production'])
    ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--local-steps',type=int,default=300)
    ap.add_argument('--factorized-rank',type=int,default=0)
    a=ap.parse_args();plan=read(a.plan)
    if a.refit:
        names=plan['development'] if a.refit=='development' else read(Path(a.dataset)/'manifest.json')['names'];val=[]
    else:
        assert a.fold is not None;names=plan['folds'][a.fold]['train'];val=plan['folds'][a.fold]['validation']
    fit(a.dataset,a.output,a.variant,a.seed,names,val,a.steps,a.local_steps,a.refit=='production',a.factorized_rank)


if __name__=='__main__':main()
