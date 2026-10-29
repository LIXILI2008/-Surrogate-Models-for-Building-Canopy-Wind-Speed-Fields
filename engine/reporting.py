"""Final all-model comparisons; routing is fitted only on original-building OOF data."""
from __future__ import annotations
from collections import defaultdict
import html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from study import ART,WORK,METHODS,SEEDS,read,write,digest
from analysis_pipeline import Labels,slice_metrics,summarize,avg

LABELS={'v23_mlp':'V2.3 MLP','unet_si':'V2.4 U-Net SI','unet':'V2.4 U-Net','fno':'V2.4 FNO',
        'ufno':'V2.4 U-FNO','mlp_data':'V2.4 MLP-data',
        'all_mean':'All-model mean','development_router':'OOF-selected model'}
COLORS=['#222222','#969696','#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']


def fit_router(p,jobs):
    old=[]
    for j in jobs:
        if j['kind']=='development':
            r=read(ART/j['key']/'evaluation.json')
            for name,b in r['buildings'].items():old.append((j['method'],name,b))
    rules={}
    for group in ['low','high']:
        for lo,hi in zip(p['height_band_edges'][:-1],p['height_band_edges'][1:]):
            band=f'{lo:g}-{hi:g}';scores={};counts={}
            # Restrict to the same observable building set for every method.
            eligible={name for method,name,b in old if method=='v23_mlp' and b['group']==group
                and b['height_bands'][band]['slices'] and b['height_bands'][band]['regions']['roof']['rmse_mps'] is not None}
            for method in METHODS:
                selected=[(name,b) for m,name,b in old if m==method and name in eligible]
                values=[b['height_bands'][band]['regions']['roof']['rmse_mps'] for _,b in selected]
                scores[method]=avg(values);counts[method]=len({b['family'] for _,b in selected})
            sufficient=all(counts[m]>=p['routing']['minimum_families'] and scores[m] is not None for m in METHODS)
            winner=min(METHODS,key=lambda m:scores[m]) if sufficient else 'v23_mlp'
            rules[f'{group}:{band}']=dict(method=winner,original_OOF_roof_rmse=scores,families=counts,
                sufficient_for_exploratory_rule=sufficient,selection_uses_external_labels=False)
    value=dict(rules=rules,source='original grouped out-of-fold predictions; one development seed',
               limitation='Small high-building development sample; this is a frozen exploratory selector, not a validated universal boundary.')
    write(ART/'routing_rule.json',value)
    return value


def route(row,p,rules):
    group='high' if row['dims'][2]>50 else 'low';r=row['z']/row['dims'][2]
    for lo,hi in zip(p['height_band_edges'][:-1],p['height_band_edges'][1:]):
        if lo<=r<hi:return rules['rules'][f'{group}:{lo:g}-{hi:g}']['method']
    raise ValueError(r)


def derived_predictions(p,rules):
    outputs={}
    for seed in SEEDS:
        original={m:np.load(ART/f'production/{m}/seed_{seed}/predictions.npy',mmap_mode='r') for m in METHODS}
        ds=Labels('production',p['external_buildings'])
        for method in ['all_mean','development_router']:
            out=ART/f'production/{method}/seed_{seed}';out.mkdir(parents=True,exist_ok=True)
            if (out/'evaluation.json').exists():
                outputs[(method,seed)]=read(out/'evaluation.json');continue
            pred=np.lib.format.open_memmap(out/'predictions.npy',mode='w+',dtype='float32',shape=original['v23_mlp'].shape)
            rows=[];selections=[]
            for i,row in enumerate(ds.rows):
                chosen=route(row,p,rules) if method=='development_router' else None
                v=original[chosen][i].copy() if chosen else np.mean([original[m][i] for m in METHODS],axis=0)
                xy,bounds,y,valid,dist=ds.get(row);pred[i]=v;rows.append(slice_metrics(row,y,v,valid,xy,dist))
                if chosen:selections.append(dict(building=row['building'],z_m=row['z'],selected=chosen))
            pred.flush();del pred
            job=dict(method=method,seed=seed,kind='production',train=p['training_buildings'],test=p['external_buildings'])
            result=summarize(rows,job,p);result['rule_sha256']=digest(ART/'routing_rule.json') if selections else None
            write(out/'per_slice.json',rows);write(out/'evaluation.json',result)
            if selections:write(out/'model_selections.json',selections)
            outputs[(method,seed)]=result
    return outputs


def production_results(p):
    out={}
    for m in METHODS+['all_mean','development_router']:
        out[m]=[read(ART/f'production/{m}/seed_{s}/evaluation.json') for s in SEEDS]
    out['inlet_baseline']=[read(ART/f'production/v23_mlp/seed_{SEEDS[0]}/baseline_evaluation.json')]
    return out


def paired_intervals(results,p):
    rng=np.random.default_rng(20261001);out={}
    for method in METHODS[1:]+['all_mean','development_router']:
        groups={}
        for group in ['all','low','high']:
            diffs=defaultdict(list)
            for name in p['external_buildings']:
                b=results['v23_mlp'][0]['buildings'][name]
                if group!='all' and b['group']!=group:continue
                a=avg([r['buildings'][name]['all_heights']['regions']['roof']['rmse_mps'] for r in results['v23_mlp']])
                v=avg([r['buildings'][name]['all_heights']['regions']['roof']['rmse_mps'] for r in results[method]])
                diffs[b['family']].append(v-a)
            families=list(diffs);draws=[]
            for _ in range(5000):
                sampled=rng.choice(families,len(families),replace=True)
                draws.append(float(np.mean([v for f in sampled for v in diffs[f]])))
            dd=[v for vv in diffs.values() for v in vv]
            groups[group]=dict(buildings=len(dd),families=len(families),mean_delta_roof_rmse_mps=float(np.mean(dd)),
                descriptive_cluster_bootstrap_95_interval=np.quantile(draws,[.025,.975]).tolist(),
                buildings_lower_error_than_MLP=sum(v<0 for v in dd))
        out[method]=groups
    write(ART/'paired_comparisons.json',dict(reference='v23_mlp',models=out,
        caveat='Descriptive intervals, not simultaneous family-wise hypothesis tests. Building-family resampling after seed averaging; pixels and heights are not independent replicates.'))
    return out


def plots(p,results):
    out=ART/'figures';out.mkdir(exist_ok=True)
    names=sorted(p['external_buildings'],key=lambda n:results['v23_mlp'][0]['buildings'][n]['label'])
    buildings=results['v23_mlp'][0]['buildings']
    a=np.array([[avg([r['buildings'][n]['all_heights']['regions']['roof']['rmse_mps'] for r in results[m]]) for m in METHODS] for n in names])
    fig,ax=plt.subplots(figsize=(13,11),layout='constrained');im=ax.imshow(a,aspect='auto',cmap='magma_r')
    ax.set_xticks(range(len(METHODS)),[LABELS[m] for m in METHODS],rotation=25,ha='right')
    ax.set_yticks(range(len(names)),[f"{buildings[n]['label']}  H={buildings[n]['dims'][2]:g}" for n in names])
    for i in range(len(names)):
        for j in range(len(METHODS)):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',color='white' if a[i,j]>np.nanmax(a)*.55 else 'black',fontsize=8)
    ax.set_title('Suzhou 20 buildings: roof RMSE, height-weighted, averaged over 3 seeds')
    fig.colorbar(im,ax=ax,label='RMSE (m/s)');fig.savefig(out/'all20_roof_rmse.png',dpi=170);plt.close(fig)
    per={m:[read(ART/f'production/{m}/seed_{s}/per_slice.json') for s in SEEDS] for m in METHODS}
    fig,axes=plt.subplots(5,4,figsize=(20,19),layout='constrained')
    for ax,n in zip(axes.flat,names):
        for m,color in zip(METHODS,COLORS):
            rr=[[r for r in rows if r['building']==n] for rows in per[m]]
            x=[r['ratio'] for r in rr[0]];y=np.array([[r['regions']['roof']['rmse_mps'] for r in rows] for rows in rr],float)
            ax.plot(x,y.mean(0),color=color,label=LABELS[m],linewidth=1)
            ax.fill_between(x,y.min(0),y.max(0),color=color,alpha=.10)
        ax.set_title(f"{buildings[n]['label']} H={buildings[n]['dims'][2]:g}m");ax.set_xlabel('z/H');ax.set_ylabel('Roof RMSE (m/s)');ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=4)
    fig.suptitle('All observed heights; line=seed mean, band=seed min/max',y=1.025)
    fig.savefig(out/'all20_error_height_curves.png',dpi=150,bbox_inches='tight');plt.close(fig)
    # Signed differences retain the pre-existing MLP versus U-FNO hypothesis, with all other methods equally visible.
    bands=[f'{lo:g}-{hi:g}' for lo,hi in zip(p['height_band_edges'][:-1],p['height_band_edges'][1:])]
    fig,axes=plt.subplots(2,3,figsize=(19,15),layout='constrained')
    diffs=[]
    for m in METHODS[1:]:
        mat=[]
        for n in names:
            vv=[]
            for b in bands:
                va=avg([r['buildings'][n]['height_bands'][b].get('regions',{}).get('roof',{}).get('rmse_mps') for r in results[m]])
                vb=avg([r['buildings'][n]['height_bands'][b].get('regions',{}).get('roof',{}).get('rmse_mps') for r in results['v23_mlp']])
                vv.append(va-vb if va is not None and vb is not None else np.nan)
            mat.append(vv)
        diffs.append(np.array(mat))
    cap=max(.01,float(np.nanmax(np.abs(np.stack(diffs)))))
    for ax,m,mat in zip(axes.flat,METHODS[1:],diffs):
        im=ax.imshow(mat,cmap='RdBu_r',vmin=-cap,vmax=cap,aspect='auto');ax.set_title(LABELS[m]+' minus V2.3 MLP')
        ax.set_xticks(range(len(bands)),bands,rotation=25);ax.set_yticks(range(len(names)),[buildings[n]['label'] for n in names]);ax.set_xlabel('z/H band')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Delta roof RMSE (m/s); blue favors V2.4 method')
    fig.savefig(out/'all20_model_height_differences.png',dpi=160);plt.close(fig)


def galleries(p, methods=None, out=None, physical_scale=True):
    methods=list(METHODS if methods is None else methods)
    ds=Labels('production',p['external_buildings']);out=ART/'gallery' if out is None else out
    out.mkdir(parents=True,exist_ok=True)
    preds={m:np.load(ART/f'production/{m}/seed_{SEEDS[0]}/predictions.npy',mmap_mode='r') for m in methods}
    reference=[read(ART/f'production/{m}/seed_{SEEDS[0]}/per_slice.json') for m in methods]
    pages=[];manifest=[]
    for name in sorted(p['external_buildings']):
        ids=[i for i,r in enumerate(ds.rows) if r['building']==name];label=ds.rows[ids[0]]['label']
        chosen=sorted({min(ids,key=lambda i:abs(ds.rows[i]['z']/ds.rows[i]['dims'][2]-r)) for r in [1.01,1.05,1.10,1.25,1.45]})
        for i in chosen:
            row=ds.rows[i];xy,bounds,y,valid,dist=ds.get(row);dims=row['dims'];z=row['z'];uin=3*(z/10)**.4
            fields=[y]+[preds[m][i] for m in methods];names=['Saved CFD reference']+[LABELS[m] for m in methods]
            fig,axes=plt.subplots(3,len(fields),figsize=(3.375*len(fields),10),layout='constrained');ims=[None]*3
            for col,(field,method) in enumerate(zip(fields,names)):
                for j,value in enumerate([field,field/uin-1,abs(field-y)]):
                    ax=axes[j,col];vmax=(6 if dims[2]<=50 else 13) if j==0 else .5 if j==1 else 2
                    extent=[-2*dims[1],3*dims[1],-1.5*dims[0],1.5*dims[0]] if physical_scale else [-2,3,-1.5,1.5]
                    if physical_scale:
                        # Treat the grid as node samples; shading='nearest' constructs half-cell edges
                        # so the physical footprint outline aligns with the sampled coordinates.
                        im=ax.pcolormesh(xy[...,0],xy[...,1],np.where(valid,value,np.nan),shading='nearest',
                            cmap=['turbo','coolwarm','magma'][j],vmin=-.5 if j==1 else 0,vmax=vmax,rasterized=True)
                        ax.set_aspect('equal',adjustable='box')
                    else:
                        im=ax.imshow(np.where(valid,value,np.nan),origin='lower',extent=extent,aspect='auto',
                                     cmap=['turbo','coolwarm','magma'][j],vmin=-.5 if j==1 else 0,vmax=vmax)
                    if physical_scale:ax.add_patch(Rectangle((-dims[1]/2,-dims[0]/2),dims[1],dims[0],fill=False,ls='--',edgecolor='black',lw=.7))
                    else:ax.add_patch(Rectangle((-.5,-.5),1,1,fill=False,ls='--',edgecolor='black',lw=.7))
                    ax.set_xticks([]);ax.set_yticks([]);ims[j]=im
                    if j==0:ax.set_title(method,fontsize=9)
                    if col==0:ax.set_ylabel(['Speed (m/s)','U/Uin-1','Absolute error (m/s)'][j])
                if col:
                    s=reference[col-1][i];roof=s['regions']['roof']['rmse_mps'];full=s['regions']['full']['rmse_mps']
                    axes[2,col].set_xlabel(f'Roof RMSE {roof:.3f}\nFull RMSE {full:.3f}',fontsize=8)
            for j in range(3):fig.colorbar(ims[j],ax=axes[j,:].tolist(),shrink=.85)
            fig.suptitle(f'{label}: W={dims[0]:.2f}, L={dims[1]:.2f}, H={dims[2]:g} m; z={z:g} m, z/H={z/dims[2]:.4f}\n'
                +('Original 28-building training; external Suzhou case; seed 20261001; axes in meters with equal physical aspect; fixed colors may saturate'
                  if physical_scale else 'Original 28-building training; external Suzhou case; seed 20261001; distorted normalized display axes x/L and y/W; fixed colors may saturate'))
            filename=f'{label}_z{z:g}.png';fig.savefig(out/filename,dpi=125);plt.close(fig)
            pages.append(f'<h2>{html.escape(label)} z={z:g} m (z/H={z/dims[2]:.4f})</h2><a href="{filename}"><img loading="lazy" width="100%" src="{filename}"></a>')
            manifest.append(dict(building=name,label=label,z_m=z,ratio=z/dims[2],image=filename,seed=SEEDS[0],
                                  valid_fraction=float(valid.mean()),display='meters; equal physical x/y aspect' if physical_scale else 'normalized axes; not equal physical x/y aspect'))
        print('gallery building complete',label,flush=True)
    heading='Suzhou 20: V2.3 MLP and five V2.4 models'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Suzhou 20 model comparisons</title><h1>'+heading+'</h1>'+''.join(pages),encoding='utf-8')
    write(out/'manifest.json',manifest)


def final_report(p,jobs):
    from cfd_summary import report as report_cfd
    report_cfd()
    rules=fit_router(p,jobs);derived_predictions(p,rules);results=production_results(p)
    from applicability import report as report_applicability
    report_applicability(p,results)
    from view_sensitivity import evaluate as evaluate_views
    evaluate_views(p)
    paired=paired_intervals(results,p);plots(p,results);galleries(p)
    groups={}
    for method,rr in results.items():
        groups[method]={g:dict(seeds=len(rr),buildings=rr[0]['groups'][g]['buildings'],
            roof_rmse_mps=avg([r['groups'][g]['regions']['roof']['rmse_mps'] for r in rr]),
            full_rmse_mps=avg([r['groups'][g]['regions']['full']['rmse_mps'] for r in rr]),
            full_r2=avg([r['groups'][g]['regions']['full']['r2'] for r in rr]),ssim=avg([r['groups'][g]['ssim'] for r in rr]),
            top10_regret_mps=avg([r['groups'][g]['screening']['top10_regret_mps'] for r in rr]),
            top10_recall=avg([r['groups'][g]['screening']['top10_recall'] for r in rr])) for g in ['all','low','high']}
    write(ART/'final_summary.json',dict(models=groups,paired_comparisons=paired,
        all_methods_complete=True,training_buildings=28,external_buildings=20,external_slices=919,
        external_untouched_blind_test=False,physical_validation='not_established',
        pending=['Representative CFD iteration stability for selected cases still below residual criteria',
                 'Additional untouched buildings required for confirmatory claims after method changes based on these 20 cases']))
    lines=['# 苏州20栋：V2.3与V2.4全部模型研究结果','',
        '统一比较六种方法：V2.3 MLP与五种V2.4配置。每种使用共同原28栋训练、3个生产种子；开发验证按足迹族分5折，另复现原49个留出评估。',
        '20栋的完整919层外部评价包含尺寸范围外样本。主表为几何归一化公共视野、建筑等权、高度间距加权。',
        '已有结果被查看，因此是外部案例验证，不宣称未查看的盲测。高组700次CFD参考及图像/数值标签差异仍限制物理解释。','',
        '| 方法 | 组别 | 栋数 | 屋顶RMSE | 全图RMSE | 全图R² | SSIM | 前10%候选速度损失 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for method,gg in groups.items():
        for group in ['low','high','all']:
            v=gg[group];nums=[v[k] for k in ['roof_rmse_mps','full_rmse_mps','full_r2','ssim','top10_regret_mps']]
            lines.append('| '+method+' | '+group+f" | {v['buildings']} | "+' | '.join('NA' if x is None else f'{x:.4f}' for x in nums)+' |')
    lines+=['','![全部20栋屋顶误差](figures/all20_roof_rmse.png)','',
        '![逐栋误差随高度变化](figures/all20_error_height_curves.png)','',
        '![各V2.4方法相对MLP的高度分组差异](figures/all20_model_height_differences.png)','',
        '[20栋多高度速度/扰动/误差图集](gallery/index.html)','',
        '[同建筑同高度视野与有效分辨率敏感性](view_sensitivity/REPORT.md)','',
        '[CFD迭代敏感性已有快照对照](cfd_sensitivity/REPORT.md)','',
        '[分区域、分高度适用范围及最低达标高度初筛](APPLICABILITY.md)','',
        '模型选择规则只根据原始建筑的分组折外评价确定，见 routing_rule.json；外部建筑CFD不参与选择。',
        '对照含所有模型等权平均和入口速度基线。配对差异的建筑族重采样区间为描述性区间，不作未经多重比较校正的显著性宣称。',
        '角流区以建筑平面四角外侧的固定几何邻域作为代理区；现有标签只有风速大小，不能据此识别旋涡或分离拓扑。网格敏感性按用户决定不做；CFD续算仅保留已完成的有限案例。',
        '少样本迁移优先候选与分组验证方案见 [TRANSFER_ADAPTATION_PLAN.md](../TRANSFER_ADAPTATION_PLAN.md)；当前报告的20栋结果仍是未适配基线，不代表迁移增益已经验证。']
    (ART/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
