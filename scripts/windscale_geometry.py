"""Read-only post-hoc diagnostics of registered V2.5 predictions; no training."""
from pathlib import Path
import argparse, ast, csv, hashlib, json, sys
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deliverables/chapter3_plot_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.transforms import Bbox

METHODS={'v23_mlp':'MLP','unet_si':'U-Net SI','unet':'U-Net','fno':'FNO','ufno':'U-FNO','mlp_data':'MLP-data'}
SEEDS=[20261001,20261002,20261003]
REGIONS=['full','roof','windward','side','wake']
BANDS=[('all',1,1.5),('1-1.02',1,1.02),('1.02-1.1',1.02,1.1),('1.1-1.25',1.1,1.25),('1.25-1.5',1.25,1.5)]
SOURCES={}
def read(p):
    p=Path(p);raw=p.read_bytes();SOURCES[str(p)]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8'))
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item()),encoding='utf-8')
def csvwrite(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ranks(a):
    a=np.asarray(a);order=np.argsort(a,kind='stable');v=a[order];out=np.empty(len(a),float)
    _,starts,counts=np.unique(v,return_index=True,return_counts=True)
    for start,count in zip(starts,counts):out[order[start:start+count]]=start+(count-1)/2+1
    return out
def corr(x,y):
    x=np.asarray(x);y=np.asarray(y)
    if np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])
def rho(x,y):return corr(ranks(x),ranks(y))
def correlation(x,y,boot=2000):
    x=np.asarray(x);y=np.asarray(y);rng=np.random.default_rng(20261016)
    vals=[]
    for ii in rng.integers(len(x),size=(boot,len(x))):
        v=rho(x[ii],y[ii])
        if v is not None:vals.append(v)
    loo=[rho(np.delete(x,i),np.delete(y,i)) for i in range(len(x))];loo=[v for v in loo if v is not None]
    return dict(rho=rho(x,y),bootstrap_ci95=np.quantile(vals,[.025,.975]).tolist(),leave_one_out_range=[min(loo),max(loo)],n=len(x))
def partial(x,y,h):
    x,y,h=map(ranks,(x,y,h));a=np.c_[np.ones(len(h)),h]
    return corr(x-a@np.linalg.lstsq(a,x,rcond=None)[0],y-a@np.linalg.lstsq(a,y,rcond=None)[0])
def savefig(out,name):
    plt.savefig(out/'figures'/f'{name}.png',dpi=300)
    plt.savefig(out/'figures'/f'{name}.svg')
    plt.close()

def annotate_points(ax,rows,feature,metric):
    ax.figure.canvas.draw();renderer=ax.figure.canvas.get_renderer();bounds=ax.get_window_extent(renderer).padded(-2)
    pts=ax.transData.transform(np.array([[r[feature],r[metric]] for r in rows]))
    marks=[Bbox.from_bounds(x-4,y-4,8,8) for x,y in pts];occupied=[]
    for i in sorted(range(len(rows)),key=lambda i:-rows[i][metric]):
        r=rows[i];best=None
        for radius in [7,13,20,29,40,55]:
            for dx,dy in [(0,radius),(0,-radius),(radius,4),(-radius,4),(radius,-7),(-radius,-7)]:
                ha='left' if dx>0 else 'right' if dx<0 else 'center';va='bottom' if dy>=0 else 'top'
                t=ax.annotate(r['id'],(r[feature],r[metric]),xytext=(dx,dy),textcoords='offset points',ha=ha,va=va,fontsize=8)
                box=t.get_window_extent(renderer).padded(1.5);t.remove()
                overlap=sum(box.overlaps(b) for b in occupied+marks);outside=not (bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1))
                score=1e6*outside+1e5*overlap+dx*dx+dy*dy
                if best is None or score<best[0]:best=(score,dx,dy,ha,va,box)
        _,dx,dy,ha,va,box=best
        ax.annotate(r['id'],(r[feature],r[metric]),xytext=(dx,dy),textcoords='offset points',ha=ha,va=va,fontsize=8,arrowprops=dict(arrowstyle='-',lw=.4,color='#888',shrinkB=4))
        occupied.append(box)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path);ap.add_argument('--output',type=Path,default=ROOT/'project_v2_5/artifacts/windscale_geometry_20261016');args=ap.parse_args()
    config=read(ROOT/'project_v2_5/config.json');base=args.results or Path(config['reference_results']);out=args.output
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    protocol=read(base/'protocol.json');old=read(ROOT/'project_v2_4/artifacts/dataset_128/manifest.json');old23=read(ROOT/'project_v2_3/artifacts/dataset_verified/manifest.json')
    frozen=read(base/'retraining_protocol.json')
    for rel,expected in frozen['original28_manifest_sha256'].items():
        path=ROOT/rel
        read(path)
        assert SOURCES[str(path)]==expected, f'Frozen data manifest changed: {path}'
    assert old['names']==old23['names'] and old['dims']==old23['dims']
    inlet=old23['config']['inlet'];assert inlet==dict(u_ref_mps=3.,z_ref_m=10.,alpha=.4)
    names=protocol['training_buildings'];assert len(names)==28 and set(names)==set(old['names'])
    lookup=dict(zip(old['names'],old['dims']));train=np.array([lookup[n] for n in names]);lo=train.min(0);hi=train.max(0);span=hi-lo
    # Execute the exact existing V2.5 weighting helpers, without importing its model/training dependencies.
    engine=ROOT/'project_v2_5/engine/analysis_pipeline.py';raw=engine.read_bytes();SOURCES[str(engine)]=hashlib.sha256(raw).hexdigest()
    tree=ast.parse(raw.decode('utf8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['height_weights','weighted']]
    env={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(engine),'exec'),env)
    weight,weighted=env['height_weights'],env['weighted']
    old_metrics=read(base/'common_evaluation/building_seed_metrics.json')
    baseline={(r['cohort'],r['method'],r['seed'],r['building']):r['regions']['full']['rmse_mps'] for r in old_metrics}
    observations=[];geom={};audit=dict(max_saved_relative_mse_difference=0.,max_baseline_rmse_difference=0.,slice_records=0,seed_runs=0)
    for cohort,folder in [('retrospective20','production'),('corrected5','corrected5')]:
        for method in METHODS:
            for seed in SEEDS:
                src=base/folder/method/f'seed_{seed}'
                slices=read(src/'per_slice.json');evaluation=read(src/'evaluation.json');ids=read(src/'prediction_rows.json')
                assert len(slices)==len(ids)
                groups=defaultdict(list)
                for s,ident in zip(slices,ids):
                    assert s['building']==ident['building'] and abs(s['z_m']-ident['z_m'])<1e-8
                    assert 1<s['ratio']<1.5 and s['building'] not in names
                    assert abs(s['z_m']/s['dims'][2]-s['ratio'])<1e-10
                    u=3*(s['z_m']/10)**.4
                    for region in REGIONS:
                        m=s['regions'][region]
                        if m['mse'] is not None:
                            diff=abs(m['relative_mse']-m['mse']/u**2)
                            audit['max_saved_relative_mse_difference']=max(audit['max_saved_relative_mse_difference'],diff)
                            assert diff<1e-10
                    groups[s['building']].append(s)
                for building,ss in groups.items():
                    label=ss[0]['label'];label='P05' if label.startswith('P05') else ({'L':'D','U':'G'}.get(label[0],label[0])+label[1:])
                    dims=ss[0]['dims'];key=(cohort,building)
                    if key in geom:assert np.allclose(geom[key]['dims'],dims)
                    geom[key]=dict(id=label,dims=dims)
                    for band,a,b in BANDS:
                        rr=sorted([s for s in ss if a<=s['ratio']<b],key=lambda s:s['ratio'])
                        if not rr:continue
                        ww=weight([s['ratio'] for s in rr],a,b)
                        for region in REGIONS:
                            mse=weighted([s['regions'][region]['mse'] for s in rr],ww)
                            norm=weighted([None if s['regions'][region]['mse'] is None else s['regions'][region]['mse']/(3*(s['z_m']/10)**.4)**2 for s in rr],ww)
                            if mse is None:continue
                            rmse=float(np.sqrt(mse));nrmse=float(np.sqrt(norm))
                            if band=='all':
                                prior=evaluation['buildings'][building]['all_heights']['regions'][region]
                                assert abs(rmse-prior['rmse_mps'])<1e-9 and abs(nrmse-prior['relative_rmse'])<1e-9
                            if band=='all' and region=='full':
                                diff=abs(rmse-baseline[(cohort,method,seed,building)])
                                audit['max_baseline_rmse_difference']=max(audit['max_baseline_rmse_difference'],diff);assert diff<1e-9
                            observations.append(dict(cohort=cohort,building=building,id=label,method=method,seed=seed,band=band,region=region,rmse_mps=rmse,nrmse_pct=100*nrmse))
                audit['slice_records']+=len(slices);audit['seed_runs']+=1
            print('diagnostics',cohort,METHODS[method],flush=True)
    # Average complete-building metrics across the same three independent training seeds.
    grouped=defaultdict(list)
    for r in observations:grouped[tuple(r[k] for k in ['cohort','building','id','method','band','region'])].append(r)
    averaged=[]
    for key,rr in grouped.items():
        assert len(rr)==3
        row=dict(zip(['cohort','building','id','method','band','region'],key))
        for metric in ['rmse_mps','nrmse_pct']:row[metric]=float(np.mean([r[metric] for r in rr]))
        averaged.append(row)
    geometry=[]
    for (cohort,building),g in geom.items():
        x=np.array(g['dims']);beyond=np.maximum(np.maximum(lo-x,x-hi),0);tol=.01
        nd=np.linalg.norm((train-x)/span,axis=1);near=int(np.argmin(nd))
        logscale=np.log(hi)-np.log(lo);ld=np.linalg.norm((np.log(train)-np.log(x))/logscale,axis=1)
        r=dict(cohort=cohort,building=building,id=g['id'],W=x[0],L=x[1],H=x[2],WH=x[0]/x[2],LH=x[1]/x[2],outside_axes=','.join(k for k,v in zip('WLH',beyond) if v>tol) or 'none',box_distance=float(np.linalg.norm(beyond/span)),nearest_distance=float(nd[near]),nearest_training=names[near],nearest_log_distance=float(ld.min()),W_exceed_m=float(beyond[0]),L_exceed_m=float(beyond[1]),H_exceed_m=float(beyond[2]))
        geometry.append(r)
    geometry.sort(key=lambda r:(r['cohort'],r['id']))
    merged=[]
    for g in geometry:
        v=next(r for r in averaged if r['building']==g['building'] and r['cohort']==g['cohort'] and r['method']=='v23_mlp' and r['band']=='all' and r['region']=='full')
        merged.append(dict(g,rmse_mps=v['rmse_mps'],nrmse_pct=v['nrmse_pct']))
    summaries=[]
    for cohort in ['retrospective20','corrected5']:
        for method in METHODS:
            for tag in ['all','low','high']:
                rr=[r for r in averaged if r['cohort']==cohort and r['method']==method and r['band']=='all' and r['region']=='full' and (tag=='all' or (geom[(cohort,r['building'])]['dims'][2]<=50)==(tag=='low'))]
                summaries.append(dict(cohort=cohort,method=method,group=tag,n=len(rr),rmse_mps=float(np.mean([r['rmse_mps'] for r in rr])),nrmse_pct=float(np.mean([r['nrmse_pct'] for r in rr]))))
    twenty=[r for r in merged if r['cohort']=='retrospective20'];five=[r for r in merged if r['cohort']=='corrected5']
    correlations=[]
    for feature in ['H','WH','LH','box_distance','nearest_distance','nearest_log_distance']:
        for metric in ['rmse_mps','nrmse_pct']:
            c=correlation([r[feature] for r in twenty],[r[metric] for r in twenty]);c.update(feature=feature,metric=metric)
            if feature!='H':c['height_adjusted_rank_correlation']=partial([r[feature] for r in twenty],[r[metric] for r in twenty],[r['H'] for r in twenty])
            correlations.append(c)
    subgroup=[]
    for label,rr in [('D',[r for r in twenty if r['H']<=50]),('G',[r for r in twenty if r['H']>50])]:
        for metric in ['rmse_mps','nrmse_pct']:
            c=correlation([r['nearest_distance'] for r in rr],[r[metric] for r in rr]);c.update(group=label,metric=metric);subgroup.append(c)
    rng=np.random.default_rng(20261016);D=[r for r in twenty if r['H']<=50];G=[r for r in twenty if r['H']>50];gap={}
    for metric in ['rmse_mps','nrmse_pct']:
        d=np.array([r[metric] for r in D]);g=np.array([r[metric] for r in G]);db=d[rng.integers(len(d),size=(5000,len(d)))].mean(1);gb=g[rng.integers(len(g),size=(5000,len(g)))].mean(1)
        gap[metric]=dict(low_mean=float(d.mean()),high_mean=float(g.mean()),high_low_ratio=float(g.mean()/d.mean()),difference=float(g.mean()-d.mean()),difference_ci95=np.quantile(gb-db,[.025,.975]).tolist(),ratio_ci95=np.quantile(gb/db,[.025,.975]).tolist())
    csvwrite(out/'building_metrics_and_geometry.csv',merged);csvwrite(out/'six_model_group_summary.csv',summaries);csvwrite(out/'seed_region_height_metrics.csv',observations)
    csvwrite(out/'building_region_height_metrics.csv',averaged)
    csvwrite(out/'training28_dimensions.csv',[dict(building=n,W=d[0],L=d[1],H=d[2]) for n,d in zip(names,train)])
    write(out/'summary.json',dict(audit=audit,training_bounds=dict(min=lo.tolist(),max=hi.tolist()),group_comparison=gap,correlations=correlations,within_group_correlations=subgroup,buildings=merged,group_summaries=summaries))
    write(out/'provenance.json',dict(project='V2.5',analysis='post-hoc; no retraining or model selection',results_root=str(base),sources=SOURCES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inlet=inlet,scope='20 primary; corrected5 displayed separately; original28 training only',normalization='100*sqrt(sum_j w_j*MSE_j/Uin(z_j)^2); per seed then building then group mean; NOT MAPE; NOT RMSE divided by a cohort speed',geometry='nearest Euclidean distance in (W,L,H), each coordinate divided by original28 training range; box distance clips to axis bounds; log-distance sensitivity; no direction swapping',statistics='20-building paired bootstrap 2000 replicates for rank correlations; stratified 5000 for high-low contrasts; correlations exploratory, not independent confirmatory tests; heights within a building not independent; seed uncertainty separate',no_source_mutations=True))
    f=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc');font_manager.fontManager.addfont(f.get_file())
    plt.rcParams.update({'font.family':f.get_name(),'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');xx=np.arange(20);col=['#2d6d9b' if r['H']<=50 else '#d58b30' for r in twenty]
    for ax,key,title in zip(axes,['rmse_mps','nrmse_pct'],['绝对误差 RMSE（m/s）','按入口风速归一化的误差（%）']):
        ax.bar(xx,[r[key] for r in twenty],color=col);ax.set_xticks(xx,[r['id'] for r in twenty],rotation=90);ax.set_ylabel(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    savefig(out,'01_absolute_and_normalized')
    for feature,fn,xlab in [('nearest_distance','02_nearest_geometry','与最近训练建筑的尺寸距离'),('box_distance','03_training_range','超出训练尺寸范围的距离')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
        for ax,metric,ylab in zip(axes,['rmse_mps','nrmse_pct'],['RMSE（m/s）','归一化误差（%）']):
            sc=ax.scatter([r[feature] for r in twenty],[r[metric] for r in twenty],c=[r['H'] for r in twenty],cmap='viridis',vmin=16,vmax=230,s=45,edgecolors='#333',lw=.5)
            ax.set(xlabel=xlab,ylabel=ylab);ax.margins(.12);ax.grid(alpha=.2)
        fig.colorbar(sc,ax=axes,label='建筑高度 H（m）',fraction=.025,pad=.02)
        for ax,metric in zip(axes,['rmse_mps','nrmse_pct']):annotate_points(ax,twenty,feature,metric)
        savefig(out,fn)
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for ax,a,b in zip(axes,[0,0,1],[2,1,2]):
        ax.scatter(train[:,a],train[:,b],c='#aaa',marker='s',s=23,label='原28栋')
        ax.scatter([r['WLH'[a]] for r in twenty],[r['WLH'[b]] for r in twenty],c='#2478aa',s=28,label='20栋')
        ax.scatter([r['WLH'[a]] for r in five],[r['WLH'[b]] for r in five],c='#d55b34',marker='^',s=38,label='修订五栋')
        ax.set(xlabel='WLH'[a]+'（m）',ylabel='WLH'[b]+'（m）');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);savefig(out,'04_geometry_coverage')
    # Reference distance need not predict an unseen building's error; no fitted selection rule.
    print(json.dumps(dict(bounds=[lo.tolist(),hi.tolist()],group_comparison=gap,correlations=correlations,buildings=merged),ensure_ascii=False,indent=2,default=lambda x:x.item()))

if __name__=='__main__':main()
