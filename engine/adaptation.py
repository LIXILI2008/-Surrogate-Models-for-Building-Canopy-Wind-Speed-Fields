"""Retrospective whole-building few-shot adaptation; frozen baseline is read-only."""
from __future__ import annotations
import os
import traceback
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from study import ART, OLD23, OLD24, EXTERNAL, METHODS, SEEDS, read, write, digest, family

OUT=ART/'adaptation_v1'
KS=[1,2,4,8]
STRATEGIES=['head','full','replay']
BANDS=[(1.,1.02),(1.02,1.1),(1.1,1.25),(1.25,1.5)]


def prepare():
    from analysis_pipeline import Labels
    if (OUT/'protocol.json').exists():
        print('Existing frozen protocol:',OUT/'protocol.json');return
    OUT.mkdir(parents=True,exist_ok=True)
    base=read(ART/'protocol.json');old=read(OLD23/'manifest.json');ext=read(EXTERNAL/'manifest.json')
    assert set(old['names'])==set(base['training_buildings'])
    assert not set(old['names'])&set(base['external_buildings'])
    assert digest(OLD23/'points.npy')==old['points_sha256']
    old24=read(OLD24/'manifest.json')
    assert set(old24['names'])==set(old['names'])
    a=dict(zip(old['names'],old['dims']));b=dict(zip(old24['names'],old24['dims']))
    assert all(np.allclose(a[n],b[n]) for n in a)
    frozen={str(ART/'protocol.json'):digest(ART/'protocol.json'),str(OLD23/'manifest.json'):digest(OLD23/'manifest.json'),str(EXTERNAL/'manifest.json'):digest(EXTERNAL/'manifest.json')}
    scores={n:[] for n in base['external_buildings']};rows={}
    for method in METHODS:
        values={n:[] for n in scores}
        for seed in SEEDS:
            directory=ART/f'production/{method}/seed_{seed}'
            frozen[str(directory/'best.pt')]=digest(directory/'best.pt')
            ev=read(directory/'evaluation.json')['buildings']
            for n in scores:
                r=ev[n];values[n].append(r['all_heights']['regions']['full']['relative_rmse'])
                rows[n]=dict(building=n,label=r['label'],dims=r['dims'],group=r['group'],family=family(r['dims']))
        for group in ['low','high']:
            ordered=sorted((n for n in scores if rows[n]['group']==group),key=lambda n:np.mean(values[n]))
            for rank,n in enumerate(ordered):scores[n].append(rank/(len(ordered)-1))
    for n in rows:
        rows[n]['difficulty']=float(np.median(scores[n]));rows[n]['residual_criteria_passed']=ext['geometry'][n]['residual_criteria_passed']
    assert len({r['family'] for r in rows.values()})==len(rows),'Group repeated footprints before splitting'
    # Main outer membership independent of observed errors; family uniqueness checked above.
    rng=np.random.default_rng(SEEDS[0]);groups=[]
    for group in ['low','high']:
        names=sorted(n for n in rows if rows[n]['group']==group)
        groups.append(list(rng.permutation(names)))
    folds=[]
    for k in range(5):
        query=[n for g in groups for n in g[2*k:2*k+2]]
        val=[n for g in groups for n in g[2*((k+1)%5):2*((k+1)%5)+2]]
        folds.append(dict(id=f'fold_{k}',kind='main_grouped_cv',query=query,validation=val,pool=sorted(set(rows)-set(query)-set(val))))
    hard=[];val=[];easy=[]
    for group in ['low','high']:
        order=sorted((n for n in rows if rows[n]['group']==group),key=lambda n:(-rows[n]['difficulty'],n))
        hard+=order[:5];val+=order[5:7];easy+=order[7:]
    folds.append(dict(id='hard_to_easy',kind='retrospective_easy_query',query=easy,validation=val,pool=hard))
    for fold in folds:
        for x,y in [('query','pool'),('query','validation'),('pool','validation')]:
            assert not {rows[n]['family'] for n in fold[x]}&{rows[n]['family'] for n in fold[y]}
        fold['supports']={}
        for seed in SEEDS:
            # Nested, difficulty-prioritized and height-group balanced. Small perturbation
            # varies support identities without drawing the easiest buildings first.
            rr=np.random.default_rng(seed);rank={n:rows[n]['difficulty']+float(rr.uniform(-.08,.08)) for n in fold['pool']}
            buckets={g:sorted([n for n in fold['pool'] if rows[n]['group']==g],key=lambda n:-rank[n]) for g in ['low','high']}
            first=max(buckets,key=lambda g:rank[buckets[g][0]]);order=[]
            for i in range(8):
                g=first if i%2==0 else ('high' if first=='low' else 'low')
                order.append(buckets[g].pop(0))
            fold['supports'][str(seed)]={str(k):order[:k] for k in KS}
    labels=Labels('production',rows);bank=[]
    rr=np.random.default_rng(SEEDS[0])
    for r in labels.rows:
        assert digest(EXTERNAL/r['cache'])==r['cache_sha256']
        xy,_,y,valid,_=labels.get(r);ix=np.flatnonzero(valid.ravel());assert len(ix)>0
        take=rr.choice(ix,4096,replace=len(ix)<4096)
        bank.append(np.column_stack((xy.reshape(-1,2)[take],y.ravel()[take]/13)).astype('float32'))
    data=OUT/'target_points';data.mkdir(exist_ok=True);np.save(data/'points.npy',np.stack(bank))
    write(data/'manifest.json',dict(records=labels.rows,points_sha256=digest(data/'points.npy'),names=sorted(rows),dims=[rows[n]['dims'] for n in sorted(rows)],sampling='4096 fixed uniform valid numeric-label pixels per slice; training half uniform half geometric strata',units='xy metres, target speed/13',source_manifest_sha256=digest(EXTERNAL/'manifest.json')))
    protocol=dict(schema=1,scope='Cross-geometry generalization and few-shot adaptation only; fixed inlet, idealized single cuboids, strict H<z<1.5H',
        interpretation='Retrospective label-informed difficulty selection. Main query memberships independent of difficulty; easy-query diagnostic is intentionally selected and cannot establish generalization on hard geometries.',
        methods=METHODS,adapted_model='v23_mlp',source_names=old['names'],seeds=SEEDS,k=KS,folds=folds,buildings=rows,
        difficulty='Median across six methods of within-height-group rank of three-seed full-field relative RMSE, measured BEFORE adaptation. Oracle difficulty, not deployable acquisition.',
        selection='At K=4 and first seed, select head/full/replay using inner-validation building-macro full-field RMSE; freeze strategy for all K and seeds in that fold. No query selection.',
        global_steps=dict(head=1500,full=1500,replay=1500,scratch=4000,joint=4000),local_steps=300,batch_points=4096,
        learning_rates=dict(head=.0001,full=.0001,replay=.0002,scratch=.0007,joint=.0007),replay_fraction=.25,replay_l2sp=.01,
        primary='Equal-building mean of height-integrated full-field RMSE in m/s; report MAE/R2/SSIM, height bands, regions, negative transfer, source-geometry support',
        frozen_baseline_sha256=frozen,target_manifest_sha256=digest(data/'manifest.json'),
        caveats=['20 targets previously viewed; not a fresh independent blind test','Original low labels are image-derived; target labels are numeric: geometry and label-domain changes are confounded','Most high cases are iteration-700 references failing residual criteria; no mesh independence claimed','Few shot means number of labelled whole buildings, with all available above-roof slices per support building','Source retention measures training-domain regression only'])
    write(OUT/'protocol.json',protocol);write(OUT/'status.json',dict(state='PREPARED',folds=6,formal_runs=216,tuning_runs=18))
    print('Prepared',len(rows),'buildings',len(labels.rows),'slices; frozen protocol',OUT/'protocol.json',flush=True)


class Engine:
    def __init__(self,p):
        import torch
        from fieldgen.train import Dataset
        from analysis_pipeline import Labels
        self.p=p;self.torch=torch;torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.source=Dataset(OLD23);self.target=Dataset(OUT/'target_points')
        self.labels=Labels('production',p['buildings']);self.cache={}
        for r in self.labels.rows:self.cache[(r['building'],r['z'])]=self.labels.get(r)

    def train(self,folder,method,names,seed):
        from fieldgen.train import load
        from fieldgen.model import Field
        from fieldgen.generalize import Sampler
        torch=self.torch;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'run.json').exists():
            model,ck=load(folder/'best.pt',self.device);return model
        torch.manual_seed(seed)
        architecture=dict(width=96,depth=4,output_mode='reference')
        source=ART/f'production/v23_mlp/seed_{seed}/best.pt'
        transfer=method in STRATEGIES
        if transfer:model,_=load(source,self.device)
        else:model=Field(**architecture).to(self.device)
        origin={n:v.detach().clone() for n,v in model.named_parameters() if not n.startswith('local.')}
        target=Sampler(self.target,names,True,seed)
        old=Sampler(self.source,self.p['source_names'],True,seed+1)
        history=[];start=time.time()
        for stage,total in [('global',self.p['global_steps'][method]),('local',self.p['local_steps'])]:
            for n,v in model.named_parameters():v.requires_grad_(n.startswith('local.') if stage=='local' else not n.startswith('local.'))
            lr=5e-5 if stage=='local' else self.p['learning_rates'][method]
            pars=[v for n,v in model.named_parameters() if n.startswith('local.')==(stage=='local')]
            optimizer=torch.optim.AdamW(pars,lr=lr,weight_decay=.0001)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total,eta_min=lr*.1)
            for step in range(1,total+1):
                if stage=='global' and method in ['head','replay']:
                    warm=method=='head' or step<=150
                    for n,v in model.named_parameters():v.requires_grad_(not n.startswith('local.') and (not warm or n.startswith(('condition.','head.'))))
                fraction=.5 if method=='joint' else self.p['replay_fraction'] if method=='replay' else 0
                nold=int(4096*fraction);ntarget=4096-nold
                ii,jj=target.indices(ntarget);batch=self.target.batch(ii,jj,self.device)
                if nold:
                    ii,jj=old.indices(nold);ob=self.source.batch(ii,jj,self.device)
                    batch=[torch.cat((a,b)) for a,b in zip(batch,ob)]
                xy,d,r,y=batch;model.train();pred,w=model(xy,d,r)
                ref=3*(d[:,2]/10).pow(.4)/13
                loss=(((pred-y)/ref).square()*w).sum()/w.sum()
                if method=='replay' and stage=='global':
                    loss+=self.p['replay_l2sp']*sum((v-origin[n]).square().mean() for n,v in model.named_parameters() if n in origin and v.requires_grad)
                if not torch.isfinite(loss):raise FloatingPointError(f'{folder}: loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step()
            history.append(dict(stage=stage,steps=total,loss=float(loss.detach()),seconds=time.time()-start))
        ck=dict(state=model.state_dict(),architecture=architecture,method=method,seed=seed,
            trained_buildings=sorted(set(names)|set(self.p['source_names']) if transfer or method=='joint' else set(names)),
            support_buildings=names,protocol_sha256=digest(OUT/'protocol.json'),source_checkpoint_sha256=digest(source) if transfer else None)
        torch.save(ck,folder/'best.pt');write(folder/'run.json',dict(method=method,seed=seed,support=names,seconds=time.time()-start,history=history,checkpoint_sha256=digest(folder/'best.pt')))
        return model.eval()

    def evaluate(self,model,names,path):
        from analysis_pipeline import infer,scalar,region_masks,aggregate_slices
        from scipy.ndimage import binary_erosion
        from skimage.metrics import structural_similarity
        if path.exists():return read(path)
        rows=[]
        for row in self.labels.rows:
            if row['building'] not in names:continue
            xy,bounds,y,valid,_=self.cache[(row['building'],row['z'])]
            p=infer(model,'v23_mlp',xy,row['dims'],row['z'],bounds,self.device)
            assert np.isfinite(p).all()
            masks=region_masks(xy,row['dims']);uin=3*(row['z']/10)**.4
            regions={k:scalar(y,p,v&valid,uin) for k,v in masks.items()}
            erosion=binary_erosion(valid,np.ones((11,11),bool),border_value=0)
            _,ss=structural_similarity(np.where(valid,y,0),np.where(valid,p,0),data_range=6 if row['dims'][2]<=50 else 13,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
            rows.append(dict(building=row['building'],ratio=row['ratio'],regions=regions,ssim=float(ss[erosion].mean()) if erosion.any() else None,screening={},roof_reference_coverage=float((masks['roof']&valid).sum()/masks['roof'].sum())))
        result={}
        for name in names:
            rr=[r for r in rows if r['building']==name]
            result[name]=dict(all_heights=aggregate_slices(rr),bands={f'{a:g}-{b:g}':aggregate_slices(rr,a,b) for a,b in BANDS})
        write(path,result);return result

    def retention(self,model,path):
        if path.exists():return
        from fieldgen.generalize import validate
        write(path,dict(interpretation='Source training-domain regression; not independent validation',result=validate(model,self.source,self.p['source_names'],self.device)))


def run_queue():
    from fieldgen.train import load
    if not (OUT/'protocol.json').exists():prepare()
    p=read(OUT/'protocol.json')
    for f,sha in p['frozen_baseline_sha256'].items():assert digest(Path(f))==sha,f
    assert digest(OUT/'target_points/manifest.json')==p['target_manifest_sha256']
    engine=Engine(p);done=0
    for fold in p['folds']:
        root=OUT/fold['id'];selection=root/'selection.json'
        if not selection.exists():
            scores={}
            for method in STRATEGIES:
                folder=root/'tuning'/method
                model=engine.train(folder,method,fold['supports'][str(SEEDS[0])]['4'],SEEDS[0])
                ev=engine.evaluate(model,fold['validation'],folder/'validation.json')
                scores[method]=float(np.mean([r['all_heights']['regions']['full']['rmse_mps'] for r in ev.values()]))
                print('TUNE',fold['id'],method,scores[method],flush=True)
            write(selection,dict(selected=min(scores,key=scores.get),validation_scores=scores,criterion=p['selection']))
        chosen=read(selection)['selected']
        for seed in SEEDS:
            base_model,_=load(ART/f'production/v23_mlp/seed_{seed}/best.pt',engine.device)
            engine.retention(base_model,OUT/f'source_retention_{seed}.json')
            for k in KS:
                support=fold['supports'][str(seed)][str(k)]
                assert not set(support)&set(fold['query']+fold['validation'])
                for route in ['adapt','scratch','joint']:
                    folder=root/f'k_{k}'/f'seed_{seed}'/route
                    model=engine.train(folder,chosen if route=='adapt' else route,support,seed)
                    engine.evaluate(model,fold['query'],folder/'query.json')
                    engine.retention(model,folder/'source_retention.json')
                    done+=1
                    write(OUT/'status.json',dict(state='RUNNING',pid=os.getpid(),updated=time.time(),completed_formal_runs=done,total_formal_runs=216,last=str(folder.relative_to(OUT))))
                    print('DONE',done,216,fold['id'],k,seed,route,flush=True)
    for f,sha in p['frozen_baseline_sha256'].items():assert digest(Path(f))==sha,f
    write(OUT/'status.json',dict(state='EXPERIMENTS_COMPLETE',completed_formal_runs=done,total_formal_runs=216,baseline_hashes_unchanged=True))
    report()


def run():
    from bench.runner import process_alive
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'runner.lock'
    if lock.exists() and process_alive(read(lock)['pid']):raise RuntimeError('Adaptation runner already active')
    write(lock,dict(pid=os.getpid(),started=time.time()))
    try:run_queue()
    except BaseException:
        write(OUT/'failure.json',dict(pid=os.getpid(),time=time.time(),error=traceback.format_exc()))
        status=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
        write(OUT/'status.json',dict(status,state='FAILED',updated=time.time()));raise
    finally:
        if lock.exists() and read(lock)['pid']==os.getpid():lock.unlink()


def report():
    p=read(OUT/'protocol.json');observations=[]
    for fold in p['folds']:
        for seed in SEEDS:
            baseline=read(ART/f'production/v23_mlp/seed_{seed}/evaluation.json')['buildings']
            for k in KS:
                # Interim tables only include matched completed triplets; no comparison
                # of the full 20-building baseline against a partly completed subset.
                pairroot=OUT/fold['id']/f'k_{k}'/f'seed_{seed}'
                if not all((pairroot/route/'query.json').exists() for route in ['adapt','scratch','joint']):continue
                for route in ['zero_shot','adapt','scratch','joint']:
                    directory=OUT/fold['id']/f'k_{k}'/f'seed_{seed}'/route
                    path=directory/'query.json'
                    if route!='zero_shot' and not path.exists():continue
                    ev=baseline if route=='zero_shot' else read(path)
                    for n in fold['query']:
                        score=ev[n]['all_heights'];base=baseline[n]['all_heights']['regions']['full']['rmse_mps']
                        d=p['buildings'][n]['dims']
                        observations.append(dict(experiment=fold['kind'],fold=fold['id'],seed=seed,k=k,route=route,building=n,label=p['buildings'][n]['label'],group=p['buildings'][n]['group'],dims=d,W_over_H=d[0]/d[2],L_over_H=d[1]/d[2],L_over_W=d[1]/d[0],difficulty=p['buildings'][n]['difficulty'],source_geometry=baseline[n]['support'],residual_criteria_passed=p['buildings'][n]['residual_criteria_passed'],rmse=score['regions']['full']['rmse_mps'],mae=score['regions']['full']['mae_mps'],r2=score['regions']['full']['r2'],ssim=score['ssim'],baseline_rmse=base,regions=score['regions'],bands=ev[n].get('height_bands',ev[n].get('bands',{}))))
    write(OUT/'observations.json',observations)
    summary=[];rng=np.random.default_rng(SEEDS[0])
    for kind in ['main_grouped_cv','retrospective_easy_query']:
        for k in KS:
            for route in ['zero_shot','adapt','scratch','joint']:
                rr=[r for r in observations if (r['experiment'],r['k'],r['route'])==(kind,k,route)]
                if not rr:continue
                by=defaultdict(list)
                for r in rr:by[r['building']].append(r)
                # Average seeds FIRST, then buildings. Uncertainty does not treat slices/seeds as independent.
                errors=np.array([np.mean([x['rmse'] for x in v]) for v in by.values()])
                changes=np.array([np.mean([x['rmse']-x['baseline_rmse'] for x in v]) for v in by.values()])
                boot=np.mean(changes[rng.integers(len(changes),size=(5000,len(changes)))],axis=1)
                summary.append(dict(experiment=kind,k=k,route=route,buildings=len(by),seed_results=len(rr),rmse=float(errors.mean()),mae=float(np.mean([r['mae'] for r in rr])),r2=float(np.mean([r['r2'] for r in rr if r['r2'] is not None])),ssim=float(np.mean([r['ssim'] for r in rr if r['ssim'] is not None])),delta_rmse=float(changes.mean()),delta_descriptive_ci95=np.quantile(boot,[.025,.975]).tolist(),negative_transfer_buildings=int((changes>0).sum())))
    write(OUT/'summary.json',summary)
    stratified=[]
    for kind in ['main_grouped_cv','retrospective_easy_query']:
        for k in KS:
            for route in ['zero_shot','adapt','scratch','joint']:
                rr=[r for r in observations if (r['experiment'],r['k'],r['route'])==(kind,k,route)]
                for group in ['all','low','high','inside_hull','inside_axes_outside_hull','outside_axis_bounds','residual_pass','residual_fail']:
                    subset=[r for r in rr if group=='all' or r['group']==group or r['source_geometry']['support_class']==group or (group=='residual_pass' and r['residual_criteria_passed']) or (group=='residual_fail' and not r['residual_criteria_passed'])]
                    for band in ['all',*[f'{a:g}-{b:g}' for a,b in BANDS]]:
                        for region in ['full','roof','windward','side','wake']:
                            by=defaultdict(list)
                            for r in subset:
                                regions=r['regions'] if band=='all' else r['bands'].get(band,{}).get('regions',{})
                                val=regions.get(region,{}).get('rmse_mps')
                                if val is not None:by[r['building']].append(val)
                            if by:stratified.append(dict(experiment=kind,k=k,route=route,group=group,band=band,region=region,buildings=len(by),rmse=float(np.mean([np.mean(v) for v in by.values()]))))
    write(OUT/'stratified.json',stratified)
    costs=[]
    for fold in p['folds']:
        for k in KS:
            for seed in SEEDS:
                for route in ['adapt','scratch','joint']:
                    folder=OUT/fold['id']/f'k_{k}'/f'seed_{seed}'/route
                    if not (folder/'source_retention.json').exists():continue
                    retained=read(folder/'source_retention.json')['result']['macro_rmse']
                    before=read(OUT/f'source_retention_{seed}.json')['result']['macro_rmse']
                    runlog=read(folder/'run.json')
                    costs.append(dict(fold=fold['id'],k=k,seed=seed,route=route,training_seconds=runlog['seconds'],source_training_domain_rmse=retained,source_regression_delta=retained-before))
    write(OUT/'cost_and_retention.json',costs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,kind,title in zip(axes,['main_grouped_cv','retrospective_easy_query'],['Grouped CV: all difficulty levels','Retrospective hard-to-easy diagnostic']):
        for route in ['zero_shot','adapt','scratch','joint']:
            rr=[r for r in summary if r['experiment']==kind and r['route']==route]
            ax.plot([r['k'] for r in rr],[r['rmse'] for r in rr],marker='o',label=route)
        ax.set(xlabel='Labelled support buildings',ylabel='Building-macro RMSE (m/s)',title=title,xticks=KS);ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'learning_curves.png',dpi=180);plt.close(fig)
    lines=['# 跨几何泛化与少样本适配','', '本实验为已查看20栋的回顾性分析。难度根据适配前六模型误差确定，不代表无需CFD标签即可选择困难样本。主分析按整栋留出覆盖全部难度；困难→容易是单独诊断，不能用来证明困难建筑泛化。','', '源模型为原28栋预训练MLP。V2.4五种配置保留统一基线，本阶段未开展其微调。每折策略在K=4、首个种子的内层验证集选定，随后固定。','', '|评价|新增栋数|方法|建筑数|RMSE m/s|R²|SSIM|较基线RMSE变化|负迁移建筑数|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['experiment']}|{r['k']}|{r['route']}|{r['buildings']}|{r['rmse']:.4f}|{r['r2']:.4f}|{r['ssim']:.4f}|{r['delta_rmse']:+.4f}|{r['negative_transfer_buildings']}|")
    lines+=['','![学习曲线](learning_curves.png)','','## 统计与范围','建筑先按高度积分，再跨种子平均，最后建筑等权。summary.json 的置信区间为按建筑重采样的描述性区间，未把切片或种子当独立样本；交叉验证训练集重叠，不能视为严格独立推断。','从头训练和联合重训4300步，适配1800步，源预训练4300步另计，因此包含计算预算差异。这里的少样本指少量完整建筑，每栋使用全部可用切片。','参考场包括未达残差标准的700次迭代结果；未完成网格独立性验证。源低组图像反解标签和新组数值标签不同，不能将差异全归因于几何。','未进行屋顶候选位置或高度初筛，也未恢复PINN、网格敏感性或CFD续算。']
    completed=len(list(OUT.glob('*/k_*/seed_*/*/query.json')))
    lines.insert(2,f'完成进度：{completed}/216 个正式训练与查询评价。'+('主分析覆盖全部20栋；困难→容易诊断另含6栋查询建筑，两者不可合并统计。' if completed==216 else '目前为中间结果，表中仅纳入四路线可配对的已完成子集。'))
    lines+=['','分区/高度/几何支持细表见 stratified.json；逐栋数据见 observations.json；训练耗时与源域保持性见 cost_and_retention.json。']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(OUT/'REPORT.md',flush=True)


def dispatch(action):
    if action=='prepare':prepare()
    elif action=='run':run()
    elif action=='report':report()
    else:print(read(OUT/'status.json') if (OUT/'status.json').exists() else {'state':'NOT_PREPARED'})
