import argparse
import copy
import time
from pathlib import Path
import numpy as np
import torch
from .common import read,write,digest
from .model import Field,Ensemble


class Dataset:
    def __init__(self,path):
        self.path=Path(path);self.meta=read(self.path/'manifest.json')
        if digest(self.path/'points.npy') != self.meta['points_sha256']:
            raise ValueError('Point supervision hash mismatch; prepare a new consistent dataset')
        self.points=np.load(self.path/'points.npy',mmap_mode='r');self.rows=self.meta['records']
        self.dims=np.array([r['dims'] for r in self.rows],np.float32)
        self.ratio=np.array([r['ratio'] for r in self.rows],np.float32)
    def batch(self,indices,point_indices,device):
        p=np.asarray(self.points[indices,point_indices]).copy()
        return [torch.as_tensor(v,device=device) for v in (p[:,:2],self.dims[indices],self.ratio[indices],p[:,2])]


def load(path,device):
    ck=torch.load(path,map_location=device,weights_only=True)
    if 'members' in ck:
        members=[]
        for member in ck['members']:
            m=Field(**member['architecture']).to(device);m.load_state_dict(member['state']);members.append(m)
        model=Ensemble(members).to(device)
    else:
        model=Field(**ck['architecture']).to(device);model.load_state_dict(ck['state'])
    model.eval()
    return model,ck


@torch.no_grad()
def evaluate(model,ds,role,device):
    names=ds.meta['splits'][role];result=[]
    for name in names:
        ids=[i for i,r in enumerate(ds.rows) if r['building']==name]
        ii=np.repeat(ids,256);jj=np.tile(np.arange(256),len(ids));sse=sae=sw=swe=base=0.;n=0
        for start in range(0,len(ii),16384):
            xy,d,r,y=ds.batch(ii[start:start+16384],jj[start:start+16384],device)
            p,w=model(xy,d,r);e=p-y
            sse+=e.square().sum().item();sae+=e.abs().sum().item();sw+=w.sum().item();swe+=(e.square()*w).sum().item()
            bg=3*(r*d[:,2]/10).pow(.4)/13;base+=(bg-y).square().sum().item();n+=len(y)
        result.append(dict(building=name,rmse=13*(sse/n)**.5,mae=13*sae/n,weighted_rmse=13*(swe/sw)**.5,background_rmse=13*(base/n)**.5,n=n))
    return dict(macro_rmse=float(np.mean([r['rmse'] for r in result])),macro_weighted_rmse=float(np.mean([r['weighted_rmse'] for r in result])),buildings=result)


def fit(dataset,output,continuity=0.,production=False,steps=None,local_steps=None):
    torch.set_num_threads(4);torch.manual_seed(20260908);rng=np.random.default_rng(20260908)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ds=Dataset(dataset);tc=ds.meta['config']['training']
    if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=True
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError('Run already exists; use a new output directory')
    architecture=dict(width=tc['width'],depth=tc['depth']);model=Field(**architecture).to(device)
    for p in model.local.parameters():p.requires_grad_(False)
    names=ds.meta['names'] if production else ds.meta['splits']['train']
    ids={name:np.array([i for i,r in enumerate(ds.rows) if r['building']==name]) for name in names}
    total=steps or tc['steps'];local_total=tc['local_steps'] if local_steps is None else local_steps
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=tc['learning_rate'],weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total,eta_min=tc['learning_rate']*.1)
    best=float('inf');history=[];start=time.time();global_step=total
    def save(stage,step,validation):
        ck=dict(state=model.state_dict(),architecture=architecture,continuity=continuity,production=production,
            trained_buildings=list(names),bounds=ds.meta['bounds'],inlet=ds.meta['config']['inlet'],palettes=ds.meta['palettes'],
            dataset_manifest_sha256=digest(ds.path/'manifest.json'),stage=stage,step=step,global_step=global_step,
            validation=validation,training_support=[dict(dims=ds.meta['dims'][ds.meta['names'].index(n)],
                ratio_max=max(ds.rows[i]['ratio'] for i in ids[n]),name=n) for n in names])
        torch.save(ck,out/'best.pt')
    for stage,limit in [('global',total),('local',local_total)]:
        if stage=='local':
            model,prior=load(out/'best.pt',device);global_step=prior['global_step']
            for name,p in model.named_parameters():p.requires_grad_(name.startswith('local.'))
            optimizer=torch.optim.AdamW(model.local.parameters(),lr=5e-5,weight_decay=.0001)
        for step in range(1,limit+1):
            model.train();chosen=rng.choice(names,size=4)
            indices=np.concatenate([np.full(tc['batch_points']//4,rng.choice(ids[n])) for n in chosen])
            pp=rng.integers(ds.points.shape[1],size=len(indices));xy,d,r,y=ds.batch(indices,pp,device)
            pred,w=model(xy,d,r);loss=((pred-y).square()*w).sum()/w.sum()
            if continuity and step%4==0:
                # Centres have labels; neighbouring heights are unlabelled prior queries.
                # Queries stay inside (H,1.5H) but may exceed this building's label coverage.
                good=(r>1.025)&(r<1.475)
                if good.any():
                    a1,_=model(xy[good],d[good],r[good]-.02);a2,_=model(xy[good],d[good],r[good]+.02)
                    loss=loss+continuity*((a1+a2-2*pred[good])/.04).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if stage=='global':scheduler.step()
            if step%tc['eval_every']==0 or step==limit:
                model.eval()
                validation=None if production else evaluate(model,ds,'validation',device)
                row=dict(stage=stage,step=step,seconds=time.time()-start,loss=loss.item(),validation=validation)
                history.append(row);write(out/'history.json',history)
                if production:
                    if step==limit:
                        if stage=='global':global_step=step
                        save(stage,step,None)
                elif validation['macro_weighted_rmse']<best:
                    # Macro building selection: all heights of a building stay out of training.
                    best=validation['macro_weighted_rmse']
                    if stage=='global':global_step=step
                    save(stage,step,validation)
                print(stage,step,'loss',round(loss.item(),6),'val',None if validation is None else round(validation['macro_weighted_rmse'],4),flush=True)
    model,ck=load(out/'best.pt',device)
    write(out/'run.json',dict(seconds=time.time()-start,continuity=continuity,production=production,
        stage=ck['stage'],global_step=ck['global_step'],local_step=ck['step'] if ck['stage']=='local' else 0,
        validation=ck['validation'],test_used_for_selection=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='artifacts/dataset');ap.add_argument('--output',required=True)
    ap.add_argument('--continuity',type=float,default=0);ap.add_argument('--production',action='store_true');ap.add_argument('--steps',type=int);ap.add_argument('--local-steps',type=int)
    a=ap.parse_args();fit(a.dataset,a.output,a.continuity,a.production,a.steps,a.local_steps)


if __name__=='__main__':main()
