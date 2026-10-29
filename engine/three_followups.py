"""Exploratory adaptation and retrospective negative-transfer conditioning."""
import argparse, os, traceback
from collections import defaultdict
import numpy as np
from study import ART,SEEDS,read,write,digest

OUT=ART/'three_followups_v1'

def conditions():
    cfg=read(ART/'sampling_comparison_v1/protocol.json');meta=read(ART/'adaptation_v1/protocol.json');rows=[]
    for j in cfg['jobs']:
        base=read(ART/f"production/v23_mlp/seed_{j['seed']}/evaluation.json")['buildings']
        ev=read(ART/f"sampling_comparison_v1/{j['fold']}/{j['policy']}/k_{j['k']}/seed_{j['seed']}/query.json")
        for n in j['query']:
            d=meta['buildings'][n]['dims'];bands={'all':ev[n]['all_heights'],**ev[n]['bands']};bb={'all':base[n]['all_heights'],**base[n]['height_bands']}
            for band,item in bands.items():
                for region in ('full','roof','wake','windward','side'):
                    a=item['regions'][region]['rmse_mps'];b=bb[band]['regions'][region]['rmse_mps']
                    if a is None or b is None:continue
                    rows.append(dict(kind=j['kind'],policy=j['policy'],k=j['k'],seed=j['seed'],building=n,band=band,region=region,
                        W_over_H=d[0]/d[2],L_over_H=d[1]/d[2],group='low' if d[2]<=50 else 'high',
                        support_class=base[n]['support']['support_class'],before=b,after=a,delta=a-b))
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in ('kind','policy','k','building','band','region'))].append(r)
    averaged=[]
    for rr in grouped.values():
        row=dict(rr[0]);row.pop('seed');row.update(before=float(np.mean([r['before'] for r in rr])),after=float(np.mean([r['after'] for r in rr])),
            delta=float(np.mean([r['delta'] for r in rr])),negative_seed_fraction=float(np.mean([r['delta']>0 for r in rr])))
        averaged.append(row)
    summary=[];rng=np.random.default_rng(20261004)
    # Geometry categories use fixed ratios, not fitted cut-points selected for a result.
    for r in averaged:
        r['W_H_bin']='<0.25' if r['W_over_H']<.25 else '0.25-1' if r['W_over_H']<1 else '>=1'
        r['L_H_bin']='<0.25' if r['L_over_H']<.25 else '0.25-1' if r['L_over_H']<1 else '>=1'
    keys=('kind','policy','k','band','region')
    for column in ('group','support_class','W_H_bin','L_H_bin'):
        groups=defaultdict(list)
        for r in averaged:groups[tuple(r[k] for k in keys)+(r[column],)].append(r)
        for key,rr in groups.items():
            delta=np.array([r['delta'] for r in rr]);boot=delta[rng.integers(0,len(rr),(2000,len(rr)))].mean(1)
            summary.append(dict(zip(keys,key[:-1]),condition=column,value=key[-1],buildings=len(rr),before=float(np.mean([r['before'] for r in rr])),
                after=float(np.mean([r['after'] for r in rr])),mean_delta=float(delta.mean()),negative_buildings=int((delta>0).sum()),
                descriptive_building_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist()))
    write(OUT/'negative_transfer/rows.json',dict(rows=averaged,summary=summary,scope='Retrospective; seeds averaged before buildings; correlated heights not independent samples; no causal attribution'))
    lines=['# 负迁移条件：K=8困难优先主分析','','全部策略/预算/高度/区域及逐栋结果见rows.json。这里展示预设高度段及几何条件。小组样本数少，不解释为可靠阈值。','',
        '|高度段|区域|条件|分组|栋数|原RMSE|适配RMSE|负迁移栋数|','|---|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['kind']=='main_grouped_cv' and r['policy']=='hard' and r['k']==8 and r['region']=='full' and (r['condition']=='group' or r['band']=='all'):
            lines.append(f"|{r['band']}|{r['region']}|{r['condition']}|{r['value']}|{r['buildings']}|{r['before']:.4f}|{r['after']:.4f}|{r['negative_buildings']}|")
    dest=OUT/'negative_transfer/REPORT.md';dest.write_text('\n'.join(lines),encoding='utf8')

def adaptation():
    import torch
    from fieldgen.train import load
    from analysis_pipeline import Labels,infer,slice_metrics,aggregate_slices
    cfg=read(ART/'sampling_comparison_v1/protocol.json');jobs=[j for j in cfg['jobs'] if j['kind']=='main_grouped_cv' and j['k']==8]
    folder=OUT/'adapted_new5';folder.mkdir(parents=True,exist_ok=True);names=read(ART/'prospective_five_v1/export/dataset/manifest.json')['names']
    registered=[]
    for j in jobs:
        ck=ART/f"sampling_comparison_v1/{j['fold']}/{j['policy']}/k_8/seed_{j['seed']}/best.pt"
        assert not set(names)&set(j['support'])
        registered.append(dict(j,checkpoint=str(ck),sha256=digest(ck)))
    protocol=dict(scope='Exploratory after original new-five results viewed; not preregistered confirmation',jobs=registered,
        choice='K8 replay fixed from old experiments; all 5 folds x 3 seeds x 3 acquisition policies; no checkpoint selection by new-five error; no new training',
        baseline='Same-seed original MLP',aggregation='Average per-checkpoint metrics over folds and seeds, then buildings; not an ensemble prediction',
        new5_excluded_from_training=True)
    if (folder/'protocol.json').exists():assert read(folder/'protocol.json')==protocol
    else:write(folder/'protocol.json',protocol)
    ds=Labels('production',names,external_root=ART/'prospective_five_v1/export/dataset');cache=[(r,*ds.get(r)) for r in ds.rows]
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    for i,j in enumerate(registered):
        dest=folder/j['policy']/j['fold']/f"seed_{j['seed']}";dest.mkdir(parents=True,exist_ok=True)
        if not (dest/'result.json').exists():
            assert digest(j['checkpoint'])==j['sha256'];model,ck=load(j['checkpoint'],device)
            assert not set(names)&set(ck['trained_buildings']);assert set(ck['support_buildings'])==set(j['support'])
            rows=[]
            for row,xy,bounds,y,valid,dist in cache:
                pred=infer(model,'v23_mlp',xy,row['dims'],row['z'],bounds,device)
                rows.append(slice_metrics(row,y,pred,valid,xy,dist))
            write(dest/'per_slice.json',rows)
            write(dest/'result.json',dict(checkpoint_sha256=j['sha256'],buildings={n:aggregate_slices([r for r in rows if r['building']==n]) for n in names}))
            del model
        write(folder/'status.json',dict(state='RUNNING',pid=os.getpid(),completed=i+1,total=len(registered)));print('ADAPTED',i+1,len(registered),flush=True)
    results=[]
    for j in registered:
        ev=read(folder/j['policy']/j['fold']/f"seed_{j['seed']}/result.json")['buildings']
        base=read(ART/f"prospective_five_v1/evaluation/v23_mlp/seed_{j['seed']}/evaluation.json")['buildings']
        for n in names:results.append(dict(policy=j['policy'],fold=j['fold'],seed=j['seed'],building=n,
            before=base[n]['all_heights']['regions']['full']['rmse_mps'],after=ev[n]['regions']['full']['rmse_mps']))
    write(folder/'observations.json',results)
    lines=['# 新五栋适配模型：探索性评价','','不含新训练；旧20栋支持集，K=8，回放策略。5个旧折×3种子权重全部评价，避免根据新五栋选择检查点。这里平均的是各模型指标，不是集成预测。','',
        '|策略|建筑|原RMSE|适配RMSE|差值|','|---|---|---:|---:|---:|']
    for policy in ('hard','random','diverse'):
        for n in names+['ALL']:
            rr=[r for r in results if r['policy']==policy and (n=='ALL' or r['building']==n)]
            a=np.mean([r['before'] for r in rr]);b=np.mean([r['after'] for r in rr]);lines.append(f'|{policy}|{n}|{a:.5f}|{b:.5f}|{b-a:+.5f}|')
    (folder/'REPORT.md').write_text('\n'.join(lines),encoding='utf8');write(folder/'status.json',dict(state='COMPLETE',completed=len(jobs),total=len(jobs)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['conditions','adaptation']);a=ap.parse_args()
    try:conditions() if a.action=='conditions' else adaptation()
    except BaseException:write(OUT/f'{a.action}_failure.json',dict(error=traceback.format_exc()));raise
