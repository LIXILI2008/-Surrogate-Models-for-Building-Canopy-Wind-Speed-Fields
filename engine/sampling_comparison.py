"""Paired support acquisition comparison; identical held-out buildings and update recipe."""
import os,time,traceback
from pathlib import Path
import numpy as np
from study import ART,SEEDS,read,write,digest
from adaptation import Engine,OUT as PRIOR,KS

OUT=ART/'sampling_comparison_v1'
POLICIES=['hard','random','diverse']

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():return read(OUT/'protocol.json')
    p=read(PRIOR/'protocol.json');jobs=[]
    for fold in p['folds']:
        pool=fold['pool'];features=np.log(np.array([p['buildings'][n]['dims'] for n in pool]))
        features=(features-features.mean(0))/np.maximum(features.std(0),1e-6)
        for seed in SEEDS:
            hard=fold['supports'][str(seed)]['8'];sequence=[p['buildings'][n]['group'] for n in hard]
            for policy in POLICIES:
                rng=np.random.default_rng(seed);chosen=[]
                for i,group in enumerate(sequence):
                    eligible=[n for n in pool if n not in chosen and p['buildings'][n]['group']==group]
                    if policy=='hard':n=hard[i]
                    elif policy=='random':n=str(rng.choice(eligible))
                    else:
                        if not chosen:n=str(rng.choice(eligible))
                        else:
                            scores={n:float(np.linalg.norm(features[pool.index(n)]-features[[pool.index(c) for c in chosen]],axis=1).min()) for n in eligible}
                            n=max(eligible,key=lambda n:scores[n])
                    chosen.append(n)
                for k in KS:
                    names=chosen[:k];assert not set(names)&set(fold['query']+fold['validation'])
                    jobs.append(dict(fold=fold['id'],kind=fold['kind'],policy=policy,seed=seed,k=k,support=names,query=fold['query'],validation=fold['validation']))
    cfg=dict(source_protocol_sha256=digest(PRIOR/'protocol.json'),update='replay',reason='Fixed original replay recipe for all selection policies; isolates selection from method tuning. Not claimed optimal.',
        geometry='Farthest-first Euclidean distance in pool-standardized log(W,L,H); random first eligible building per seed.',
        balance='All policies share the same ordered low/high group sequence as frozen hard selection, nested K.',
        caveats=['Hard difficulty is retrospectively label-informed','Same previously viewed query buildings; no fresh blind test','Three seeds only; uncertainty descriptive','Geometry features and recipe fixed before this comparison; do not select on queries'],jobs=jobs)
    write(OUT/'protocol.json',cfg);return cfg

def report(cfg):
    from collections import defaultdict
    rows=[]
    for j in cfg['jobs']:
        folder=OUT/j['fold']/j['policy']/f'k_{j["k"]}'/f'seed_{j["seed"]}'
        if not (folder/'query.json').exists():continue
        ev=read(folder/'query.json');base=read(ART/f'production/v23_mlp/seed_{j["seed"]}/evaluation.json')['buildings']
        for n in j['query']:
            rows.append(dict(kind=j['kind'],policy=j['policy'],k=j['k'],seed=j['seed'],building=n,rmse=ev[n]['all_heights']['regions']['full']['rmse_mps'],baseline=base[n]['all_heights']['regions']['full']['rmse_mps']))
    write(OUT/'observations.json',rows);lines=['# 固定查询集的选样对照','','更新方法统一为带源数据回放的MLP微调，1500+300步。三策略除支持建筑身份外条件相同。主分析与容易查询诊断分开。','','|评价|K|策略|建筑数|RMSE m/s|原模型RMSE|负迁移栋数|','|---|---:|---|---:|---:|---:|---:|']
    for kind in ['main_grouped_cv','retrospective_easy_query']:
        for k in KS:
            for policy in POLICIES:
                rr=[r for r in rows if (r['kind'],r['k'],r['policy'])==(kind,k,policy)];by=defaultdict(list)
                for r in rr:by[r['building']].append(r)
                if not by:continue
                a=[np.mean([r['rmse'] for r in v]) for v in by.values()];b=[np.mean([r['baseline'] for r in v]) for v in by.values()]
                lines.append(f'|{kind}|{k}|{policy}|{len(by)}|{np.mean(a):.4f}|{np.mean(b):.4f}|{sum(x>y for x,y in zip(a,b))}|')
    lines+=['','完成前的策略间样本覆盖可能不同，中间表不用于策略排序。最终仅按相同建筑、种子、K配对比较。']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    from bench.runner import process_alive
    cfg=prepare();lock=OUT/'runner.lock'
    if lock.exists() and process_alive(read(lock)['pid']):raise RuntimeError('Already running')
    write(lock,dict(pid=os.getpid(),started=time.time()))
    try:
        p=read(PRIOR/'protocol.json')
        for f,sha in p['frozen_baseline_sha256'].items():assert digest(Path(f))==sha
        engine=Engine(p)
        for i,j in enumerate(cfg['jobs']):
            folder=OUT/j['fold']/j['policy']/f'k_{j["k"]}'/f'seed_{j["seed"]}'
            folder.mkdir(parents=True,exist_ok=True);write(folder/'acquisition.json',dict(j,protocol_sha256=digest(OUT/'protocol.json')))
            model=engine.train(folder,'replay',j['support'],j['seed'])
            engine.evaluate(model,j['query'],folder/'query.json');engine.retention(model,folder/'source_retention.json')
            write(OUT/'status.json',dict(state='RUNNING',pid=os.getpid(),completed=i+1,total=len(cfg['jobs']),last=j,updated=time.time()))
            print('DONE',i+1,len(cfg['jobs']),j['fold'],j['policy'],j['k'],j['seed'],flush=True)
        report(cfg)
        for f,sha in p['frozen_baseline_sha256'].items():assert digest(Path(f))==sha
        write(OUT/'status.json',dict(state='COMPLETE',completed=len(cfg['jobs']),total=len(cfg['jobs']),baseline_hashes_unchanged=True))
    except BaseException:
        write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
    finally:
        if lock.exists() and read(lock)['pid']==os.getpid():lock.unlink()

if __name__=='__main__':main()
