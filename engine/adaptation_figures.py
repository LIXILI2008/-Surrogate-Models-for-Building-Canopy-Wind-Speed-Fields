"""Matched whole-fold interim diagnostics and equal-scale physical field panels."""
from collections import defaultdict
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from study import ART, SEEDS, read, write, digest
from adaptation import OUT, KS


def main():
    from analysis_pipeline import Labels, infer, scalar
    from fieldgen.train import load
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    protocol=read(OUT/'protocol.json');dest=OUT/'diagnostics';dest.mkdir(exist_ok=True)
    complete=[]
    for fold in protocol['folds']:
        if fold['kind']!='main_grouped_cv':continue
        if all((OUT/fold['id']/f'k_{k}'/f'seed_{s}'/route/'query.json').exists() for k in KS for s in SEEDS for route in ['adapt','scratch','joint']):complete.append(fold)
    names=[n for f in complete for n in f['query']]
    baseline={s:read(ART/f'production/v23_mlp/seed_{s}/evaluation.json')['buildings'] for s in SEEDS}
    rows=[]
    for fold in complete:
        for k in KS:
            for route in ['adapt','scratch','joint']:
                evaluations={s:read(OUT/fold['id']/f'k_{k}'/f'seed_{s}'/route/'query.json') for s in SEEDS}
                for n in fold['query']:
                    old=float(np.mean([baseline[s][n]['all_heights']['regions']['full']['rmse_mps'] for s in SEEDS]))
                    new=float(np.mean([evaluations[s][n]['all_heights']['regions']['full']['rmse_mps'] for s in SEEDS]))
                    rows.append(dict(building=n,label=protocol['buildings'][n]['label'],fold=fold['id'],k=k,route=route,baseline_rmse=old,rmse=new,gain_mps=old-new))
    write(dest/'building_gains.json',dict(completed_folds=[f['id'] for f in complete],buildings=len(names),rows=rows))
    ordered=sorted(names,key=lambda n:protocol['buildings'][n]['label'])
    if ordered:
        fig,axes=plt.subplots(1,3,figsize=(12,max(4,len(ordered)*.3)),layout='constrained')
        lim=max(abs(r['gain_mps']) for r in rows) or 1
        for ax,route in zip(axes,['adapt','scratch','joint']):
            matrix=np.array([[next(r['gain_mps'] for r in rows if r['building']==n and r['k']==k and r['route']==route) for k in KS] for n in ordered])
            im=ax.imshow(matrix,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
            ax.set(xticks=range(4),xticklabels=KS,yticks=range(len(ordered)),yticklabels=[protocol['buildings'][n]['label'] for n in ordered],xlabel='Support buildings',title=route)
        fig.colorbar(im,ax=axes,label='RMSE reduction (m/s); positive = improvement')
        fig.savefig(dest/'building_gains.png',dpi=180);plt.close(fig)
    labels=Labels('production',names);entries=[]
    seed=SEEDS[0]
    zero,_=load(ART/f'production/v23_mlp/seed_{seed}/best.pt',device)
    for fold in complete:
        models={'zero_shot':zero};paths={'zero_shot':ART/f'production/v23_mlp/seed_{seed}/best.pt'}
        for route in ['adapt','scratch','joint']:
            path=OUT/fold['id']/'k_8'/f'seed_{seed}'/route/'best.pt'
            models[route],_=load(path,device);paths[route]=path
        for name in fold['query']:
            candidates=[r for r in labels.rows if r['building']==name]
            for requested in [1.05,1.25]:
                row=min(candidates,key=lambda r:abs(r['ratio']-requested));label=protocol['buildings'][name]['label']
                target=dest/f'{label}_r{requested:.2f}.png';meta=target.with_suffix('.json')
                entries.append(dict(label=label,image=target.name,z_m=row['z'],ratio=row['ratio'],fold=fold['id']))
                if target.exists() and meta.exists() and read(meta).get('layout_version')==2:continue
                xy,bounds,y,valid,_=labels.get(row);w,l,h=row['dims']
                predictions={r:infer(m,'v23_mlp',xy,row['dims'],row['z'],bounds,device) for r,m in models.items()}
                assert all(np.isfinite(v).all() for v in predictions.values())
                fields={'CFD reference':y,**predictions};vmax=6 if h<=50 else 13
                fig,axes=plt.subplots(2,5,figsize=(18,max(4,6*(3*w/(5*l))+1.8)),layout='constrained')
                metrics={}
                for col,(route,field) in enumerate(fields.items()):
                    speed=axes[0,col].pcolormesh(xy[...,0],xy[...,1],np.ma.array(field,mask=~valid),shading='auto',cmap='turbo',vmin=0,vmax=vmax)
                    error=axes[1,col].pcolormesh(xy[...,0],xy[...,1],np.ma.array(abs(field-y),mask=~valid),shading='auto',cmap='inferno',vmin=0,vmax=2)
                    score=scalar(y,field,valid,3*(row['z']/10)**.4);metrics[route]=score
                    axes[0,col].set_title(route)
                    axes[1,col].set_title(f"RMSE {score['rmse_mps']:.3f} m/s")
                    for ax in axes[:,col]:
                        ax.add_patch(Rectangle((-l/2,-w/2),l,w,fill=False,linestyle='--',color='black'))
                        ax.set(aspect='equal',xlabel='x (m)',ylabel='y (m)',xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]))
                fig.colorbar(speed,ax=axes[0,:],label='Speed (m/s)',extend='max')
                fig.colorbar(error,ax=axes[1,:],label='Absolute error (m/s)',extend='max')
                fig.suptitle(f'{label}: W={w:.3f}, L={l:.3f}, H={h:g} m; z={row["z"]:g} m, z/H={row["ratio"]:.4f}\nWhole-building query; K=8; seed={seed}; equal x/y metre scale; +x inflow; white=invalid; colors saturate')
                fig.savefig(target,dpi=150);plt.close(fig)
                write(meta,dict(layout_version=2,building=name,dims_WLH_m=row['dims'],z=row['z'],requested_ratio=requested,actual_ratio=row['ratio'],seed=seed,k=8,fold=fold['id'],checkpoint_sha256={r:digest(p) for r,p in paths.items()},slice_metrics=metrics))
            print('FIGURES',label,flush=True)
    parts=['<!doctype html><meta charset="utf-8"><title>少样本适配：整栋留出图集</title><style>body{font-family:sans-serif;max-width:1500px;margin:30px auto}img{width:100%;height:auto}article{margin:40px 0}</style>',f'<h1>少样本适配：{len(names)}栋、{len(entries)}张对比图</h1><p>只纳入全部K、种子、路线已完成的主分析折。固定展示K=8、首种子及最接近z/H=1.05、1.25的实有切片，不按效果挑图。图中x/y按米等比例；色限饱和不改变数值指标。中间结果不能代表最终20栋。</p>']
    if ordered:parts.append('<h2>逐栋误差变化</h2><img src="building_gains.png">')
    for e in entries:parts.append(f'<article><h2>{html.escape(e["label"])}：z={e["z_m"]:g} m，z/H={e["ratio"]:.4f}</h2><a href="{e["image"]}"><img loading="lazy" src="{e["image"]}"></a></article>')
    (dest/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    write(dest/'manifest.json',dict(completed_folds=[f['id'] for f in complete],buildings=len(names),figures=len(entries),entries=entries))
    print(dest/'index.html',flush=True)


if __name__=='__main__':main()
