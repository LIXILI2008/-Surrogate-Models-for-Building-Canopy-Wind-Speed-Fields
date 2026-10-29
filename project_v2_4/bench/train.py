import argparse
import time
import os
import numpy as np
import torch
from torch.nn import functional as F
from .common import ROOT,read,write,digest
from .data import Dataset,flat_sample
from .models import make,CoordinateModel
from .geometry import features
from .physics import residuals


def supervised(p,b,si_ablation=False):
    d=b['target'][...,0];valid=b['valid'];feat=b['f'];gate=feat[...,14]
    # Bounded per-class weights are computed solely from this training batch.
    weight=(1+6*gate)*valid
    if not si_ablation:
        cls=torch.where(d<=-.1,0,torch.where(d>=.1,2,1))
        for k in range(3):
            mask=(cls==k)&valid
            count=mask.sum(-1,keepdim=True)
            balance=(valid.sum(-1,keepdim=True)/(3*count.clamp_min(1))).clamp(.5,5)
            weight=torch.where(mask,weight*balance,weight)
    err=p[...,0]-d
    if si_ablation:
        uin=3*(b['dims'][...,2]*b['q'][...,2]/10)**.4
        err=err*uin/13
    loss=(err.square()*weight).sum()/weight.sum().clamp_min(1)
    if not si_ablation:
        # Soft region supervision; thresholds fixed before evaluating any new model.
        c1=F.binary_cross_entropy_with_logits((-p[...,0]-.1)/.04,(d<=-.1).float(),reduction='none')
        c2=F.binary_cross_entropy_with_logits((p[...,0]-.1)/.04,(d>=.1).float(),reduction='none')
        loss=loss+.002*((c1+c2)*weight).sum()/weight.sum().clamp_min(1)
    mask=b['aux_valid']
    # Identical auxiliary-label access for every architecture, including the no-PDE control.
    if mask.any():
        aux=((p[...,1:5]-b['target'][...,1:5]).square().mean(-1))[mask].mean()
        kt=(torch.asinh(p[...,5:]/torch.tensor([.01,.001],device=p.device))-
            torch.asinh(b['target'][...,5:]/torch.tensor([.01,.001],device=p.device))).square().mean(-1)
        loss=loss+.1*aux+.01*kt[mask].mean()
    return loss


def physics_loss(model,ds,names,rng,device,count=96):
    high=[n for n in names if ds.dimensions[n][2]>50];name=str(rng.choice(high))
    row=ds.rows[int(rng.choice(ds.ids[name]))];w,l,h=row['dims'];x0,y0,x1,y1=row['bounds']
    # Interior box only; keep a >=2m wall distance, not Fluent wall-function cells.
    q=np.stack((rng.uniform(x0/h,x1/h,count*3),rng.uniform(y0/h,y1/h,count*3),rng.uniform(1.001,1.499,count*3)),-1)
    clearance=np.sqrt(np.maximum(abs(q[:,0])*h-l/2,0)**2+np.maximum(abs(q[:,1])*h-w/2,0)**2+((q[:,2]-1)*h)**2)
    q=q[clearance>2][:count]
    assert len(q)==count
    qt=torch.tensor(q,dtype=torch.float32,device=device,requires_grad=True)
    dims=torch.tensor(row['dims'],dtype=torch.float32,device=device).expand(count,3)
    span=torch.tensor(row['span'],dtype=torch.float32,device=device).expand(count,2)
    output=model(qt,dims,span);r=residuals(output,qt,dims)
    # Each nondimensional equation is logged separately; no hidden dimensional scaling.
    losses=r.square().mean(0)
    return losses.mean(),losses.detach().cpu().tolist()


def save_checkpoint(path,ck):
    tmp=path.with_suffix('.tmp.pt');torch.save(ck,tmp)
    for attempt in range(100):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)


def fit(dataset,out,method,held_out,steps=3000,local_steps=300,seed=20260911):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32=False # stable AD/metric precision across fresh processes
    ds=Dataset(dataset);names=sorted(set(ds.meta['names'])-{held_out})
    assert held_out not in names and held_out in ds.ids
    architecture='unet' if method=='unet_si' else method;model=make(architecture).to(device)
    rng=np.random.default_rng(seed);rp=np.random.default_rng(seed+1);started=time.time();history=[]
    signature=dict(training_protocol=2,method=method,held_out=held_out,seed=seed,steps=steps,local_steps=local_steps,
                   dataset_manifest_sha256=digest(dataset/'manifest.json'))
    if (out/'run.json').exists():
        assert read(out/'run.json')['signature']==signature;return
    resume=None
    if (out/'resume.pt').exists():
        resume=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        assert resume['signature']==signature;model.load_state_dict(resume['state'])
        rng.bit_generator.state=resume['rng'];rp.bit_generator.state=resume['physics_rng'];history=resume['history']
    elapsed=resume.get('elapsed',0) if resume else 0
    try:
        for stage,total in [('global',steps),('local',local_steps)]:
            if resume and stage=='global' and resume['stage']=='local':continue
            for n,p in model.named_parameters():p.requires_grad_(n.startswith('local.') if stage=='local' else not n.startswith('local.'))
            lr=5e-5 if stage=='local' else 5e-4
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(total,1),eta_min=lr*.1)
            start=0
            if resume and resume['stage']==stage:
                optimizer.load_state_dict(resume['optimizer']);scheduler.load_state_dict(resume['scheduler']);start=resume['step'];resume=None
            for step in range(start+1,total+1):
                model.train();ids=ds.sample(names,rng);batch=ds.batch(ids,device)
                idx=rng.integers(ds.meta['size']**2,size=(2,4096));sample=flat_sample(batch,idx)
                if isinstance(model,CoordinateModel):p=model.from_features(sample['f'])
                else:
                    full=model(batch['f'].permute(0,3,1,2));index=torch.tensor(idx,device=device)
                    p=full.flatten(2).transpose(1,2).gather(1,index[...,None].expand(-1,-1,7))
                loss=supervised(p,sample,method=='unet_si');data_loss=float(loss.detach());phys=None
                if method=='pinn' and stage=='global' and step>200:
                    pl,phys=physics_loss(model,ds,names,rp,device)
                    loss=loss+.001*min(1,(step-200)/800)*pl
                # Observed roof no-slip on auxiliary velocities, common to all methods when present.
                roof=sample['f'][...,10]>.5
                if stage=='global' and roof.any():loss=loss+.02*p[...,1:4][roof].square().mean()
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss {method} {step}')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();scheduler.step()
                if step%100==0 or step==total:
                    item=dict(stage=stage,step=step,loss=float(loss.detach()),data_loss=data_loss,pde_mse=phys,
                              seconds=elapsed+time.time()-started,gradient_norm=float(norm))
                    history.append(item);write(out/'history.json',history)
                    write(out/'status.json',dict(state='TRAINING',method=method,held_out=held_out,**item))
                    print(method,held_out,stage,step,round(item['loss'],6),flush=True)
                if step%250==0 or step==total:
                    save_checkpoint(out/'resume.pt',dict(signature=signature,state=model.state_dict(),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),stage=stage,step=step,rng=rng.bit_generator.state,
                        physics_rng=rp.bit_generator.state,history=history,elapsed=elapsed+time.time()-started))
        ck=dict(state={k:v.cpu() for k,v in model.state_dict().items()},architecture=architecture,signature=signature,
                trained_buildings=names,held_out=held_out,source='V2.4 controlled multi-output comparison',
                scalar_target='D=speed/Uin(z)-1',pde_weight=.001 if method=='pinn' else 0,
                parameter_count=sum(p.numel() for p in model.parameters()),bounds=ds.meta['bounds'])
        save_checkpoint(out/'best.pt',ck)
        write(out/'run.json',dict(signature=signature,seconds=elapsed+time.time()-started,parameters=ck['parameter_count'],
            checkpoint_sha256=digest(out/'best.pt'),trained_buildings=names,held_out_used_in_training=False,
            supervised_points_per_step=8192,physics_points_per_step=96 if method=='pinn' else 0,
            gpu=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0))
        write(out/'status.json',dict(state='TRAINED',method=method,held_out=held_out))
    except BaseException:
        import traceback
        write(out/'status.json',dict(state='FAILED',error=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--held-out',required=True)
    p.add_argument('--dataset',default=str(ROOT/'artifacts/dataset_128'));p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--local-steps',type=int,default=300);a=p.parse_args()
    from pathlib import Path
    fit(Path(a.dataset),Path(a.output),a.method,a.held_out,a.steps,a.local_steps)
