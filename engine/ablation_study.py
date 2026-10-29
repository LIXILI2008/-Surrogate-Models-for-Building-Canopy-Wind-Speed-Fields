"""Original-28 grouped mechanism ablations and matched replay/cost audit."""
import argparse,os,time,traceback,copy
from pathlib import Path
import numpy as np
from study import ART,OLD23,SEEDS,read,write,digest,family
from fieldgen.model import Field
from fieldgen.geometry import features
import torch

OUT=ART/'ablation_v1'
VARIANTS=['baseline','no_sdf_features','no_condition','direct_speed']

class AblatedField(Field):
    def __init__(self,variant='baseline'):
        super().__init__(width=96,depth=4,output_mode='reference');self.variant=variant
    def forward(self,xy,dims,ratio):
        if self.variant=='baseline':return super().forward(xy,dims,ratio)
        f,c,bg,gate,roof,weight=features(xy,dims,ratio)
        if self.variant=='no_sdf_features':
            f=f.clone();f[:,4:6]=0
        if self.variant=='no_condition':c=torch.zeros_like(c)
        scale,shift=self.condition(c).chunk(2,-1)
        t=torch.nn.functional.silu(self.input(f)*(1+.2*scale.tanh())+.2*shift)
        for layer in self.layers:t=torch.nn.functional.silu(layer(t)+.1*t)
        factor=self.head(t)[:,0]+gate*self.local(torch.cat((f,c),-1))[:,0]
        if self.variant=='direct_speed':
            # Positive direct normalized speed; fixed 4 m/s initialization, no U(z) addition.
            initial=4/13;offset=initial+np.log(-np.expm1(-5*initial))/5
            pred=torch.nn.functional.softplus(factor+offset,beta=5)
        else:
            ref=3*(dims[:,2]/10).pow(.4)/13;base=bg/ref
            inv=base+torch.log(-torch.expm1(-5*base))/5
            pred=ref*torch.nn.functional.softplus(inv+factor,beta=5)
        return torch.where(roof,torch.zeros_like(pred),pred),weight

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():return read(OUT/'protocol.json')
    jobs=[j for j in read(ART/'jobs.json') if j['kind']=='development' and j['method']=='v23_mlp']
    meta=read(OLD23/'manifest.json');dd=dict(zip(meta['names'],meta['dims']));folds=[]
    for j in jobs:
        train=j['train'];groups={}
        for n in train:groups.setdefault(family(dd[n]),[]).append(n)
        # Select two distinct footprint families by geometry, without outcome labels.
        ordered=sorted(groups,key=lambda g:np.mean([dd[n][2] for n in groups[g]]))
        chosen=[ordered[0],ordered[-1]];support=[n for g in chosen for n in groups[g]]
        source=sorted(set(train)-set(support));test=j['test']
        assert not {family(dd[n]) for n in source}&{family(dd[n]) for n in support+test}
        folds.append(dict(id=j['key'].split('/')[-1],train=train,query=test,source=source,support=support))
    p=dict(scope='Original 28 only. Same five footprint-family development folds, three seeds. No new-five or Suzhou query tuning.',
        variants=VARIANTS,folds=folds,seeds=SEEDS,steps=4000,local_steps=300,batch=4096,
        definitions=dict(no_sdf_features='Only spatial feature channels sdf/H and sdf3/H zeroed. Geometry-dependent coordinates, masks, sampling, gate and weights retained.',no_condition='Five explicit size/aspect condition inputs zeroed in global and local branches; geometry in coordinates/SDF remains.',direct_speed='Direct positive speed/13 with fixed 4m/s initialization; no analytic U(z) background addition. Same loss U(H) normalization retained.'),
        replay='Within each outer training fold reserve two complete footprint families as target supports. Source pretraining excludes target supports and queries. Paired adaptation with 0 or 25% source replay, same 1500+300 steps, LR .0002, warmup150 and L2SP .01. Replay changes allocation within fixed4096 batch; query untouched.',
        cost='Serialize after sampling queue. Record training-only wall time with CUDA sync, hardware and same-grid warmed inference. CFD runtime cannot be recovered from iteration count; user must record new simulation time.',
        hashes={str(OLD23/'manifest.json'):digest(OLD23/'manifest.json'),str(Path(__file__)):digest(Path(__file__))})
    write(OUT/'protocol.json',p);return p

def train(ds,names,variant,seed,folder,source_ck=None,replay_names=None,replay_fraction=0):
    from fieldgen.generalize import Sampler
    folder.mkdir(parents=True,exist_ok=True);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AblatedField(variant).to(device)
    if (folder/'run.json').exists():
        ck=torch.load(folder/'best.pt',weights_only=False,map_location=device);model.load_state_dict(ck['state']);return model.eval()
    torch.manual_seed(seed);model=AblatedField(variant).to(device)
    if source_ck:model.load_state_dict(torch.load(source_ck,map_location=device,weights_only=False)['state'])
    origin={n:p.detach().clone() for n,p in model.named_parameters() if not n.startswith('local.')}
    target=Sampler(ds,names,True,seed);old=Sampler(ds,replay_names,True,seed+1) if replay_names else None
    dims=dict(zip(ds.meta['names'],ds.meta['dims']));low=[n for n in names if dims[n][2]<=50];high=[n for n in names if dims[n][2]>50]
    a=Sampler(ds,low,True,seed+1) if low else None;b=Sampler(ds,high,True,seed+2) if high else None
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    for stage,total in [('global',1500 if source_ck else 4000),('local',300)]:
        for n,p in model.named_parameters():p.requires_grad_(n.startswith('local.') if stage=='local' else not n.startswith('local.'))
        lr=5e-5 if stage=='local' else .0002 if source_ck else .0007
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total,eta_min=lr*.1)
        for step in range(total):
            if source_ck and stage=='global':
                for n,p in model.named_parameters():p.requires_grad_(not n.startswith('local.') and (step>=150 or n.startswith(('condition.','head.'))))
            if source_ck and replay_fraction:
                nold=int(4096*replay_fraction);i,j=target.indices(4096-nold);ii,jj=old.indices(nold);i=np.r_[i,ii];j=np.r_[j,jj]
            elif not source_ck and a and b:
                i,j=a.indices(2048);ii,jj=b.indices(2048);i=np.r_[i,ii];j=np.r_[j,jj]
            else:i,j=target.indices(4096)
            xy,d,r,y=ds.batch(i,j,device);model.train();pred,w=model(xy,d,r);ref=3*(d[:,2]/10).pow(.4)/13
            loss=(((pred-y)/ref).square()*w).sum()/w.sum()
            if source_ck and stage=='global':loss+=.01*sum((p-origin[n]).square().mean() for n,p in model.named_parameters() if n in origin and p.requires_grad)
            if not torch.isfinite(loss):raise FloatingPointError(folder)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();scheduler.step()
    if device=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    torch.save(dict(state=model.state_dict(),variant=variant,trained_buildings=names,source_checkpoint=str(source_ck) if source_ck else None,seed=seed,protocol_sha256=digest(OUT/'protocol.json')),folder/'best.pt')
    write(folder/'run.json',dict(training_seconds=seconds,device=torch.cuda.get_device_name(0) if device=='cuda' else 'cpu',variant=variant,seed=seed,source_ck_sha256=digest(source_ck) if source_ck else None,replay_fraction=replay_fraction,parameter_count=sum(p.numel() for p in model.parameters())))
    return model.eval()

def evaluate(model,names,path):
    from analysis_pipeline import Labels,infer,slice_metrics,aggregate_slices
    if path.exists():return
    ds=Labels('development',names);rows=[];device=next(model.parameters()).device
    for r in ds.rows:
        xy,bounds,y,valid,dist=ds.get(r);v=infer(model,'v23_mlp',xy,r['dims'],r['z'],bounds,device)
        row=slice_metrics(r,y,v,valid,xy,dist);row['screening']={};rows.append(row)
    results={n:aggregate_slices([r for r in rows if r['building']==n]) for n in names}
    r=ds.rows[0];xy,bounds,*_=ds.get(r)
    for _ in range(3):infer(model,'v23_mlp',xy,r['dims'],r['z'],bounds,device)
    if device.type=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(10):infer(model,'v23_mlp',xy,r['dims'],r['z'],bounds,device)
    if device.type=='cuda':torch.cuda.synchronize()
    write(path,dict(buildings=results,inference_seconds_per_128_grid=(time.perf_counter()-start)/10))

def report(p):
    lines=['# 原28栋最小消融与成本比较','','相同开发折、相同三种子；逐栋等权。SDF/条件消融仍保留其他几何来源，不称完全无几何模型。','', '|变体|已完成折种子数|整场RMSE m/s|训练秒均值|','|---|---:|---:|---:|']
    for variant in VARIANTS:
        vals={};times=[];count=0
        for fold in p['folds']:
            for seed in SEEDS:
                root=OUT/fold['id']/f'seed_{seed}'/variant
                if not (root/'query.json').exists():continue
                count+=1;times.append(read(root/'run.json')['training_seconds'])
                for n,r in read(root/'query.json')['buildings'].items():vals.setdefault(n,[]).append(r['regions']['full']['rmse_mps'])
        if vals:lines.append(f'|{variant}|{count}/15|{np.mean([np.mean(v) for v in vals.values()]):.4f}|{np.mean(times):.2f}|')
    lines+=['','|更新方式|完成折种子数|查询整场RMSE|源训练域RMSE|更新训练秒|','|---|---:|---:|---:|---:|']
    for route in ['source','no_replay','replay']:
        vals={};ret=[];times=[];count=0
        for fold in p['folds']:
            for seed in SEEDS:
                root=OUT/fold['id']/f'seed_{seed}'/route
                if not (root/'source_retention.json').exists() or not (root/'query.json').exists():continue
                count+=1;ret.append(read(root/'source_retention.json')['macro_rmse']);times.append(read(root/'run.json')['training_seconds'])
                for n,r in read(root/'query.json')['buildings'].items():vals.setdefault(n,[]).append(r['regions']['full']['rmse_mps'])
        if vals:lines.append(f'|{route}|{count}/15|{np.mean([np.mean(v) for v in vals.values()]):.4f}|{np.mean(ret):.4f}|{np.mean(times):.2f}|')
    lines+=['','回放对照的 query.json 为外层未见建筑；source_retention.json 为源训练域保持性，不能称独立泛化。全部完成前不根据部分折排序。source行是预训练成本，适配行仅计更新成本；与baseline的全部外层训练数据从头训练比较时须计入源预训练成本。','CFD求解耗时、网格耗时与人工操作成本需分别记录；不能用700次迭代推算小时数。']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    from bench.runner import process_alive
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int);args=ap.parse_args();p=prepare()
    write(OUT/'status.json',dict(state='WAITING_FOR_GPU',pid=os.getpid(),wait_pid=args.wait_pid))
    if args.wait_pid:
        while process_alive(args.wait_pid):time.sleep(5)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    from fieldgen.train import Dataset
    from fieldgen.generalize import validate
    ds=Dataset(OLD23);done=0
    try:
        for fold in p['folds']:
            for seed in SEEDS:
                root=OUT/fold['id']/f'seed_{seed}'
                for variant in VARIANTS:
                    model=train(ds,fold['train'],variant,seed,root/variant);evaluate(model,fold['query'],root/variant/'query.json');done+=1
                    write(OUT/'status.json',dict(state='RUNNING',pid=os.getpid(),completed_training_tasks=done,total=105,last=str(root/variant)))
                src=root/'source';base=train(ds,fold['source'],'baseline',seed,src);evaluate(base,fold['query'],src/'query.json');done+=1
                write(src/'source_retention.json',validate(base,ds,fold['source'],next(base.parameters()).device))
                for route,fraction in [('no_replay',0),('replay',.25)]:
                    folder=root/route;model=train(ds,fold['support'],'baseline',seed,folder,src/'best.pt',fold['source'],fraction)
                    evaluate(model,fold['query'],folder/'query.json')
                    write(folder/'source_retention.json',validate(model,ds,fold['source'],next(model.parameters()).device));done+=1
                    write(OUT/'status.json',dict(state='RUNNING',pid=os.getpid(),completed_training_tasks=done,total=105,last=str(folder)))
                report(p)
        write(OUT/'status.json',dict(state='COMPLETE',completed_training_tasks=done,total=105));report(p)
    except BaseException:
        write(OUT/'status.json',dict(state='FAILED',completed=done,error=traceback.format_exc()));raise

if __name__=='__main__':main()
