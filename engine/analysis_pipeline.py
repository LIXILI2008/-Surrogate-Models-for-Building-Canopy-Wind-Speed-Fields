"""Matched grids, all-height diagnostics and external screening for six methods."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import shutil
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_erosion
from scipy.spatial import Delaunay
from skimage.metrics import structural_similarity

from study import ROOT,WORK,ART,OLD23,OLD24,EXTERNAL,METHODS,SEEDS,family,read,write,digest


def avg(xs):
    xs=[float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def grid(dims,size=128):
    w,l,h=dims;bounds=[-2*l,-1.5*w,3*l,1.5*w]
    x,y=np.meshgrid(np.linspace(bounds[0],bounds[2],size),np.linspace(bounds[1],bounds[3],size))
    return np.stack([x,y],-1).astype('float32'),bounds


def region_masks(xy,dims):
    w,l,h=dims;x,y=xy[...,0],xy[...,1];ax,ay=abs(x),abs(y)
    roof=(ax<=l/2)&(ay<=w/2)
    edge_width=.1*min(w,l)
    core=(ax<=l/2-edge_width)&(ay<=w/2-edge_width)
    # Speed-only labels cannot locate vortices. This frozen geometric proxy isolates
    # the four exterior plan-corner neighborhoods for a corner-flow-associated audit.
    corner_width=max(1.,.1*min(w,l))
    corner_flow_proxy=(ax>=l/2)&(ax<=l/2+corner_width)&(ay>=w/2)&(ay<=w/2+corner_width)
    return dict(full=np.ones(x.shape,bool),roof=roof,roof_core=core,roof_edge=roof&~core,
        windward=(x < -l/2)&(x>=-l/2-h)&(ay<=w/2),
        corner_flow_proxy=corner_flow_proxy,
        side=(ax<=l/2)&(ay>w/2)&(ay<=w/2+min(h,w)),
        wake=(x>l/2)&(x<=l/2+2*h)&(ay<=max(w,.25*h)))


def audit(p):
    old=read(OLD24/'manifest.json');ext=read(EXTERNAL/'manifest.json')
    for path,sha in p['dataset_manifest_sha256'].items():
        assert digest(WORK/path)==sha, path
    assert digest(OLD23/'points.npy')==read(OLD23/'manifest.json')['points_sha256']
    for f,sha in old['array_sha256'].items(): assert digest(OLD24/f)==sha,f
    unique={};rows=[];seen=set()
    for r in ext['records']:
        assert r['building'] in p['external_buildings'] and 1<r['z']/r['dims'][2]<1.5
        key=(r['building'],round(r['z'],4));assert key not in seen;seen.add(key)
        assert digest(EXTERNAL/r['cache'])==r['cache_sha256'],r['cache']
        unique[r['source']]=r['source_sha256']
    for k,(src,sha) in enumerate(unique.items()):
        assert digest(src)==sha,src
        if (k+1)%100==0:print('audit raw source',k+1,len(unique),flush=True)
    for name in sorted(p['external_buildings']):
        rr=[r for r in ext['records'] if r['building']==name];g=ext['geometry'][name]
        d=g['dims'];boundary=[]
        for j,k in enumerate('WLH'):
            if d[j]<old['bounds']['min'][j] or d[j]>old['bounds']['max'][j]:boundary.append(k)
        rows.append(dict(building=name,label=g['label'],dims_WLH_m=d,slices=len(rr),
            z_min=min(r['z'] for r in rr),z_max=max(r['z'] for r in rr),
            ratio_min=min(r['ratio'] for r in rr),ratio_max=max(r['ratio'] for r in rr),
            outside_axes=boundary,case=g['case'],data=g['data'],residual_criteria_passed=g['residual_criteria_passed'],
            source_label='native Fluent node values',inlet_max_relative_error=max(r['inlet_boundary_relative_error'] for r in rr)))
    result=dict(state='DATA_AND_SPLIT_AUDIT_PASSED',training_buildings=28,external_buildings=20,
        external_slices=len(ext['records']),raw_sources_verified=len(unique),external_rows=rows,
        training_label_domains=read(OLD23/'manifest.json').get('label_domains'),
        cache_only_physical_validation=False,
        caveats=['All 20 external buildings previously viewed; not untouched blind testing.',
                 'Original low group uses image-derived labels; external low group uses numerical Fluent labels.',
                 'High external group uses iteration 700; most residual criteria not satisfied.',
                 'Geometry-scaled view is a change from legacy image crop; evaluate this transfer explicitly.'])
    write(ART/'audit.json',result)
    print('data audit complete',len(rows),'buildings',flush=True)
    reproduce_history()


def reproduce_history():
    from bench.evaluate import evaluate
    from bench.runner import report
    orig=WORK/'project_v2_4/artifacts/comparison';p=read(orig/'plan.json');out=ART/'historical'
    differences=[]
    for method in ['v23_reference',*p['methods']]:
        for case in p['cases']:
            name=case['name'];dest=out/method/name;dest.mkdir(parents=True,exist_ok=True)
            if method!='v23_reference' and not (dest/'best.pt').exists():
                shutil.copy2(orig/method/name/'best.pt',dest/'best.pt')
            new=evaluate(OLD24,dest,name,legacy=method=='v23_reference')
            previous=read(orig/method/name/'metrics.json')
            delta={k:float(new['speed'][k]-previous['speed'][k]) for k in ['rmse','mae','r2']}
            differences.append(dict(method=method,building=name,difference=delta,
                                    matches=all(abs(v)<1e-5 for v in delta.values())))
            print('historical reproduced',method,name,flush=True)
    report(out,p)
    write(out/'reproduction_audit.json',dict(runs=len(differences),all_match=all(r['matches'] for r in differences),rows=differences))
    if not all(r['matches'] for r in differences):raise ValueError('Historical reproduction differs; inspect audit before proceeding')


def load_model(job):
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu';path=ART/job['key']/'best.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert set(ck['trained_buildings'])==set(job['train'])
    assert not set(ck['trained_buildings'])&set(job['test'])
    if job['method']=='v23_mlp':
        from fieldgen.train import load
        model,ck=load(path,device)
    else:
        from bench.models import make
        model=make(ck['architecture']).to(device);model.load_state_dict(ck['state']);model.eval()
    return model,ck,device


def infer(model,method,xy,dims,z,bounds,device):
    import torch
    from bench.geometry import features
    from bench.models import CoordinateModel
    if method=='v23_mlp':
        from fieldgen.predict import infer as infer23
        return infer23(model,xy,dims,z,device)
    q=np.concatenate([xy/dims[2],np.full((*xy.shape[:-1],1),z/dims[2],dtype='float32')],-1)
    qt=torch.tensor(q,device=device);dt=torch.tensor(dims,device=device,dtype=torch.float32).expand_as(qt)
    st=torch.tensor([bounds[2]-bounds[0],bounds[3]-bounds[1]],device=device,dtype=torch.float32).expand(*qt.shape[:-1],2)
    with torch.no_grad():
        f=features(qt,dt,st)
        if isinstance(model,CoordinateModel):
            chunks=[model.from_features(x)[:,0].cpu().numpy() for x in f.reshape(-1,17).split(8192)]
            v=np.concatenate(chunks).reshape(xy.shape[:-1])
        else:v=model(f.permute(2,0,1)[None])[0,0].cpu().numpy()
    return (v+1)*float(3*(z/10)**.4)


class Labels:
    def __init__(self,kind,names,external_root=None):
        self.kind=kind
        self.external_root=Path(external_root) if external_root is not None else EXTERNAL
        self.meta=read((self.external_root if kind=='production' else OLD24)/'manifest.json')
        self.rows=[]
        for i,r in enumerate(self.meta['records']):
            if r['building'] in names and 1+1e-7<r['z']/r['dims'][2]<1.5-1e-7:
                self.rows.append(dict(r,source_index=i))
        self.rows.sort(key=lambda r:(r['building'],r['z']))
        if kind!='production':
            self.targets=np.load(OLD24/'targets.npy',mmap_mode='r');self.valid=np.load(OLD24/'valid.npy',mmap_mode='r')
        self.grids={}

    def get(self,row):
        n=row['building']
        if n not in self.grids:self.grids[n]=grid(row['dims'])
        xy,bounds=self.grids[n]
        if self.kind=='production':
            assert np.allclose(bounds,row['bounds'])
            with np.load(self.external_root/row['cache']) as a:
                y=a['speed_mps'].copy();valid=a['valid'].copy();dist=a['nearest_node_m'].copy()
        else:
            x0,y0,x1,y1=row['bounds'];xx=np.linspace(x0,x1,128);yy=np.linspace(y0,y1,128);i=row['source_index']
            q=xy[...,[1,0]];uin=3*(row['z']/10)**.4
            raw=(np.asarray(self.targets[i,0])+1)*uin
            y=RegularGridInterpolator((yy,xx),raw,bounds_error=False,fill_value=np.nan)(q).astype('float32')
            valid=RegularGridInterpolator((yy,xx),np.asarray(self.valid[i],dtype='float32'),bounds_error=False,fill_value=0)(q)>.999
            valid &= np.isfinite(y);y=np.where(valid,y,0).astype('float32');dist=None
        return xy,bounds,y,valid,dist


def scalar(y,p,mask,uin):
    if not mask.any():return dict(n=0,rmse_mps=None,mae_mps=None,r2=None,mse=None,mean_y=None,mean_y2=None,relative_mse=None)
    a=y[mask].astype('float64');b=p[mask].astype('float64');e=b-a;mse=float(np.mean(e*e));v=float(np.var(a))
    return dict(n=int(mask.sum()),rmse_mps=mse**.5,mae_mps=float(np.mean(abs(e))),r2=1-mse/v if v>1e-12 else None,
                mse=mse,mean_y=float(a.mean()),mean_y2=float(np.mean(a*a)),relative_mse=mse/uin**2)


def screening(y,p,mask):
    if mask.sum()<10:return dict(top10_recall=None,top10_regret_mps=None,predicted_top10_mean=None,selected_true_mean=None,true_top10_mean=None)
    yy=y[mask].astype('float64');pp=p[mask].astype('float64');k=max(1,int(np.ceil(len(yy)*.1)))
    truth=np.argsort(yy,kind='stable')[-k:];chosen=np.argsort(pp,kind='stable')[-k:]
    return dict(top10_recall=len(np.intersect1d(truth,chosen))/k,
        top10_regret_mps=float(yy[truth].mean()-yy[chosen].mean()),predicted_top10_mean=float(pp[chosen].mean()),
        selected_true_mean=float(yy[chosen].mean()),true_top10_mean=float(yy[truth].mean()))


def slice_metrics(row,y,p,valid,xy,dist=None):
    if not np.isfinite(p).all():raise FloatingPointError(f"Nonfinite prediction {row['building']} z={row['z']}")
    masks=region_masks(xy,row['dims']);uin=float(3*(row['z']/10)**.4)
    regions={k:scalar(y,p,v&valid,uin) for k,v in masks.items()}
    ss=None;eroded=binary_erosion(valid,np.ones((11,11),bool),border_value=0)
    if eroded.any():
        _,image=structural_similarity(np.where(valid,y,0),np.where(valid,p,0),data_range=6 if row['dims'][2]<=50 else 13,
            gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
        ss=float(image[eroded].mean())
    roof=masks['roof'];coverage=float((roof&valid).sum()/roof.sum())
    candidate=screening(y,p,roof&valid) if coverage>=.99 else screening(y,p,np.zeros_like(valid))
    return dict(building=row['building'],z_m=row['z'],ratio=row['z']/row['dims'][2],height_above_roof_m=row['z']-row['dims'][2],
        label=row.get('label',row['building']),dims=row['dims'],regions=regions,ssim=ss,screening=candidate,
        roof_reference_coverage=coverage,full_reference_coverage=float(valid.mean()),
        nearest_node_roof_p95_m=float(np.quantile(dist[roof&valid],.95)) if dist is not None and (roof&valid).any() else None,
        predicted_min_mps=float(p.min()),predicted_max_mps=float(p.max()))


def height_weights(ratios,lo=1.,hi=1.5):
    rr=np.asarray(ratios,dtype=float)
    if not len(rr):return np.array([])
    assert np.all(np.diff(rr)>0)
    edge=np.r_[lo,(rr[:-1]+rr[1:])/2,hi]
    widths=np.maximum(0,np.diff(np.clip(edge,lo,hi)))
    return widths/widths.sum()


def weighted(values,weights):
    ii=[i for i,v in enumerate(values) if v is not None and np.isfinite(v)]
    total=sum(weights[i] for i in ii)
    return float(sum(float(values[i])*weights[i] for i in ii)/total) if total>0 else None


def aggregate_slices(rows,lo=1.,hi=1.5):
    rr=sorted([r for r in rows if lo<=r['ratio']<hi],key=lambda r:r['ratio'])
    if not rr:return dict(slices=0,regions={},ssim=None,screening={})
    w=height_weights([r['ratio'] for r in rr],lo,hi);regions={}
    for region in rr[0]['regions']:
        mm=[r['regions'][region] for r in rr]
        mse=weighted([m['mse'] for m in mm],w);mu=weighted([m['mean_y'] for m in mm],w);mu2=weighted([m['mean_y2'] for m in mm],w)
        variance=None if mu is None else max(0,mu2-mu*mu)
        rel=weighted([m['relative_mse'] for m in mm],w)
        regions[region]=dict(rmse_mps=mse**.5 if mse is not None else None,
            mae_mps=weighted([m['mae_mps'] for m in mm],w),
            r2=1-mse/variance if variance is not None and variance>1e-12 else None,
            mean_slice_r2=weighted([m['r2'] for m in mm],w),relative_rmse=rel**.5 if rel is not None else None)
    return dict(slices=len(rr),ratio_min=rr[0]['ratio'],ratio_max=rr[-1]['ratio'],
                max_ratio_gap=float(np.diff([r['ratio'] for r in rr]).max()) if len(rr)>1 else None,
                regions=regions,ssim=weighted([r['ssim'] for r in rr],w),
                screening={k:weighted([r['screening'][k] for r in rr],w) for k in rr[0]['screening']},
                roof_reference_coverage_min=min(r['roof_reference_coverage'] for r in rr))


def height_screening(rows,thresholds):
    rr=sorted(rows,key=lambda r:r['z_m']);out={}
    for v in thresholds:
        truth=[r for r in rr if r['screening']['true_top10_mean'] is not None and r['screening']['true_top10_mean']>=v]
        chosen=[r for r in rr if r['screening']['predicted_top10_mean'] is not None and r['screening']['predicted_top10_mean']>=v]
        a=truth[0] if truth else None;b=chosen[0] if chosen else None
        out[str(v)]=dict(true_lowest_z_m=a['z_m'] if a else None,predicted_lowest_z_m=b['z_m'] if b else None,
            height_error_m=b['z_m']-a['z_m'] if a and b else None,missed_feasible=bool(a and not b),
            false_recommendation=bool(b and b['screening']['selected_true_mean']<v),
            selected_true_speed_mps=b['screening']['selected_true_mean'] if b else None,
            reference_roof_coverage_complete=all(r['roof_reference_coverage']>=.99 for r in rr))
    return out


def geometry_support(dims,train):
    meta=read(OLD24/'manifest.json');dd=dict(zip(meta['names'],meta['dims']));a=np.array([dd[n] for n in train]);x=np.asarray(dims)
    lo=a.min(0);hi=a.max(0);scale=np.maximum(hi-lo,1e-8);inside=bool(np.all(x>=lo-.01)&np.all(x<=hi+.01))
    try:hull=bool(Delaunay((a-lo)/scale).find_simplex((x-lo)/scale,tol=1e-7)>=0)
    except Exception:hull=None
    return dict(inside_axis_bounds=inside,inside_geometry_hull=hull,nearest_normalized_distance=float(np.linalg.norm((a-x)/scale,axis=1).min()),
                support_class='outside_axis_bounds' if not inside else 'inside_axes_outside_hull' if not hull else 'inside_hull')


def summarize(rows,job,p):
    buildings={};edges=p['height_band_edges']
    for name in sorted({r['building'] for r in rows}):
        rr=[r for r in rows if r['building']==name];dims=rr[0]['dims']
        bands={f'{lo:g}-{hi:g}':aggregate_slices(rr,lo,hi) for lo,hi in zip(edges[:-1],edges[1:])}
        near=[r for r in rr if 0<r['height_above_roof_m']<=1.00001]
        buildings[name]=dict(label=rr[0]['label'],dims=dims,family=family(dims),group='high' if dims[2]>50 else 'low',
            all_heights=aggregate_slices(rr),height_bands=bands,
            first_meter=aggregate_slices(near,1,min(1.5,1+1/dims[2]+1e-7)),
            support=geometry_support(dims,job['train']),height_screening=height_screening(rr,p['screening']['absolute_speed_thresholds_mps']))
    groups={}
    for g in ['all','low','high','inside_hull','inside_axes_outside_hull','outside_axis_bounds']:
        bb=[b for b in buildings.values() if g=='all' or b['group']==g or b['support']['support_class']==g]
        if not bb:continue
        groups[g]=dict(buildings=len(bb),regions={r:{k:avg([b['all_heights']['regions'][r][k] for b in bb])
                      for k in ['rmse_mps','mae_mps','r2','mean_slice_r2','relative_rmse']} for r in p['regions']},
                      ssim=avg([b['all_heights']['ssim'] for b in bb]),
                      screening={k:avg([b['all_heights']['screening'][k] for b in bb]) for k in ['top10_recall','top10_regret_mps']})
    return dict(method=job['method'],kind=job['kind'],seed=job['seed'],buildings=buildings,groups=groups,
                slices=len(rows),physical_validation='not_established',scope='agreement with saved CFD on geometry-scaled common grid')


def evaluate_job(job,p):
    out=ART/job['key'];ds=Labels(job['kind'],job['test'],external_root=job.get('external_root'));model,ck,device=load_model(job)
    pred=np.lib.format.open_memmap(out/'predictions.npy',mode='w+',dtype='float32',shape=(len(ds.rows),128,128))
    rows=[];base=[];started=time.time()
    for i,row in enumerate(ds.rows):
        xy,bounds,y,valid,dist=ds.get(row)
        v=infer(model,job['method'],xy,row['dims'],row['z'],bounds,device).astype('float32')
        pred[i]=v;rows.append(slice_metrics(row,y,v,valid,xy,dist))
        if job['method']=='v23_mlp' and job['seed']==SEEDS[0]:
            base.append(slice_metrics(row,y,np.full_like(y,3*(row['z']/10)**.4),valid,xy,dist))
        if (i+1)%100==0 or i+1==len(ds.rows):
            write(out/'evaluation_status.json',dict(done=i+1,total=len(ds.rows),seconds=time.time()-started))
            print('evaluation',job['key'],i+1,len(ds.rows),flush=True)
    pred.flush();del pred
    write(out/'prediction_rows.json',[dict(building=r['building'],z_m=r['z'],source_index=r['source_index']) for r in ds.rows])
    write(out/'per_slice.json',rows)
    result=summarize(rows,job,p);result.update(checkpoint_sha256=digest(out/'best.pt'),prediction_sha256=digest(out/'predictions.npy'),seconds=time.time()-started,
        evaluation_code_sha256=digest(Path(__file__)))
    write(out/'evaluation.json',result)
    if base:
        baseline=summarize(base,dict(job,method='inlet_baseline'),p)
        write(out/'baseline_evaluation.json',baseline);write(out/'baseline_per_slice.json',base)


def progress_report(p,jobs):
    complete=[];trained=[]
    for j in jobs:
        path=ART/j['key']
        if (path/'run.json').exists():trained.append(j['key'])
        if (path/'evaluation.json').exists():complete.append(dict(job=j,result=read(path/'evaluation.json')))
    write(ART/'progress.json',dict(total_training_jobs=len(jobs),trained=len(trained),evaluated=len(complete),
          completed_keys=[r['job']['key'] for r in complete],updated=time.time()))
    lines=['# 苏州20栋原始V2.3与V2.4全模型研究','',f'训练完成 {len(trained)}/{len(jobs)}；评价完成 {len(complete)}/{len(jobs)}。',
        '', '统一比较：V2.3 MLP；V2.4 U-Net SI、U-Net、FNO、U-FNO、MLP-data。',
        '全部模型只训练原始建筑；苏州20栋全部用于外部评价，包括13栋超出旧单轴尺寸范围的案例。',
        '当前表格随已完成任务更新，不能把未齐全的种子/模型当作最终排名。',
        '参考是保存的CFD解；高组700次、CFD物理可靠性尚未建立。所有评价为公共几何视野128×128。','',
        '| 方法 | 完成生产种子 | 低组屋顶RMSE | 高组屋顶RMSE | 全组候选速度损失 |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        rr=[r['result'] for r in complete if r['job']['kind']=='production' and r['job']['method']==method]
        vals=[avg([r['groups'][g]['regions']['roof']['rmse_mps'] for r in rr]) for g in ['low','high']]
        vals+=[avg([r['groups']['all']['screening']['top10_regret_mps'] for r in rr])]
        lines.append('| '+method+' | '+str(len(rr))+' | '+' | '.join('待完成' if v is None else f'{v:.4f}' for v in vals)+' |')
    lines+=['','低/高组分别10栋；先按高度间距汇总每栋，再建筑等权，最后对已完成的种子取均值。',
            '历史复现见 historical/RESULTS.md；逐栋、逐高度、区域指标和筛选结果见各任务的 evaluation.json 与 per_slice.json。',
            '源数据审计见 audit.json；任务队列见 jobs.json；运行状态见 status.json。']
    (ART/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return complete


def finalize(p,jobs):
    complete=progress_report(p,jobs)
    if len(complete)!=len(jobs):raise RuntimeError('All models, seeds and folds must finish before final analysis')
    from reporting import final_report
    final_report(p,jobs)
