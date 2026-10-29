"""Predefined frame/resolution transfer diagnostic at the same external heights."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import LinearNDInterpolator,RegularGridInterpolator
from study import ART,EXTERNAL,METHODS,SEEDS,read,write
from analysis_pipeline import Labels,load_model,infer,region_masks,scalar,avg


def evaluate(p, methods=None, out=None):
    methods=list(METHODS if methods is None else methods)
    out=ART/'view_sensitivity' if out is None else out
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():return read(out/'results.json')
    protocol=dict(seed=SEEDS[0],ratios=[1.01,1.05,1.10,1.25,1.45],legacy_span_m=72.35,
        methods=methods,training_changed=False,
        comparison='Same source z, common physical query coordinates and joint valid mask. Geometry-frame predictions are bilinearly sampled onto legacy query grid. This measures combined frame and effective-resolution transfer, not pure architecture.')
    write(out/'protocol.json',protocol)
    ds=Labels('production',p['external_buildings']);prepared=[]
    x,y=np.meshgrid(np.linspace(-36.175,36.175,128),np.linspace(-36.175,36.175,128));xy=np.stack([x,y],-1).astype('float32')
    bounds=[-36.175,-36.175,36.175,36.175]
    for name in p['external_buildings']:
        ids=[i for i,r in enumerate(ds.rows) if r['building']==name]
        chosen=sorted({min(ids,key=lambda i:abs(ds.rows[i]['z']/ds.rows[i]['dims'][2]-q)) for q in protocol['ratios']})
        for i in chosen:
            row=ds.rows[i];path=row['source'];query=xy+np.asarray(row['center'])
            with np.load(path) as a:
                coords=np.column_stack([a['x-coordinate'],a['y-coordinate']]).astype(float);values=a['velocity-magnitude'].astype(float)
            unique,inv=np.unique(np.round(coords,7),axis=0,return_inverse=True)
            value=np.bincount(inv,weights=values)/np.bincount(inv)
            target=np.asarray(LinearNDInterpolator(unique,value)(query),dtype='float32')
            primary_xy,pbounds,_,primary_valid,_=ds.get(row)
            sx=np.linspace(pbounds[0],pbounds[2],128);sy=np.linspace(pbounds[1],pbounds[3],128)
            supported=RegularGridInterpolator((sy,sx),primary_valid.astype(float),bounds_error=False,fill_value=0)(xy[...,[1,0]])>.999
            valid=np.isfinite(target)&supported
            prepared.append(dict(i=i,row=row,target=target,valid=valid,sx=sx,sy=sy))
    results=[]
    for method in methods:
        job=dict(kind='production',key=f'production/{method}/seed_{SEEDS[0]}',method=method,seed=SEEDS[0],train=p['training_buildings'],test=p['external_buildings'])
        model,ck,device=load_model(job);pred=np.load(ART/job['key']/'predictions.npy',mmap_mode='r')
        for item in prepared:
            row=item['row'];v=infer(model,method,xy,row['dims'],row['z'],bounds,device)
            primary=RegularGridInterpolator((item['sy'],item['sx']),pred[item['i']],bounds_error=False,fill_value=np.nan)(xy[...,[1,0]])
            valid=item['valid']&np.isfinite(primary);masks=region_masks(xy,row['dims']);uin=3*(row['z']/10)**.4
            w,l,h=row['dims'];coverage=min(72.35,l)*min(72.35,w)/(w*l)
            a={k:scalar(item['target'],primary,valid&mask,uin) for k,mask in masks.items()}
            b={k:scalar(item['target'],v,valid&mask,uin) for k,mask in masks.items()}
            results.append(dict(method=method,building=row['building'],label=row['label'],z_m=row['z'],ratio=row['z']/h,
                legacy_footprint_geometric_coverage=coverage,common_valid_fraction=float(valid.mean()),
                geometry_frame_on_common_grid=a,legacy_frame_on_common_grid=b))
        del model,pred
        print('view sensitivity complete',method,flush=True)
    result=dict(protocol=protocol,rows=results)
    write(out/'results.json',result)
    lines=['# 视野与有效空间分辨率敏感性','',protocol['comparison'],'',
           '每栋使用预定5个相对高度附近的实际源层，仅用种子20261001。所有模型固定同一训练权重。',
           '旧72.35m视野不能覆盖最长体块的完整足迹；这些行只作共同可见区域诊断，不解释为完整屋顶指标。','',
           '| 模型 | 宽视野→共同网格屋顶RMSE | 旧视野屋顶RMSE |','|---|---:|---:|']
    for m in methods:
        rr=[r for r in results if r['method']==m and r['legacy_footprint_geometric_coverage']>=.99]
        by={n:[r for r in rr if r['building']==n] for n in sorted({r['building'] for r in rr})}
        a=avg([avg([r['geometry_frame_on_common_grid']['roof']['rmse_mps'] for r in rows]) for rows in by.values()])
        b=avg([avg([r['legacy_frame_on_common_grid']['roof']['rmse_mps'] for r in rows]) for rows in by.values()])
        lines.append(f'| {m} | {a:.4f} | {b:.4f} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
