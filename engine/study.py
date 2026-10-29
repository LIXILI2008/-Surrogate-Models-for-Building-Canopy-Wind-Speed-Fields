"""Frozen original-version study. Never trains on the Suzhou external cohort."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
ART = Path(os.environ.get('WINDFIELD_V25_RESULTS', str(WORK / 'artifacts')))
DATA_WORKSPACE = Path('C:\\Users\\lxl\\Documents\\ChatGPT\\new2')
OLD23 = DATA_WORKSPACE / 'project_v2_3/artifacts/dataset_verified'
OLD24 = DATA_WORKSPACE / 'project_v2_4/artifacts/dataset_128'
EXTERNAL = DATA_WORKSPACE / 'project_roof_unified/artifacts/data'
METHODS = ['v23_mlp', 'unet_si', 'unet', 'fno', 'ufno', 'mlp_data']
SEEDS = [20261001, 20261002, 20261003]
for path in (WORK/'project_v2_3', WORK/'project_v2_4'):
    sys.path.insert(0, str(path))
from bench.common import read, write, digest


def family(d):
    return '_'.join(f'{x:.1f}' for x in sorted(d[:2]))


def source_files():
    return sorted([*list((WORK/'project_v2_3/fieldgen').glob('*.py')),
                   *list((WORK/'project_v2_4/bench').glob('*.py'))])


def generate_trainers():
    """Preserve original recipes; replace only train membership and provenance."""
    src24 = WORK/'project_v2_4/bench/train.py'
    s = src24.read_text(encoding='utf-8')
    edits = [
        ('from .', 'from bench.'),
        ('def fit(dataset,out,method,held_out,steps=3000,local_steps=300,seed=20260911):',
         'def fit(dataset,out,method,held_out=None,steps=3000,local_steps=300,seed=20260911,training_names=None):'),
        ("ds=Dataset(dataset);names=sorted(set(ds.meta['names'])-{held_out})\n    assert held_out not in names and held_out in ds.ids",
         "ds=Dataset(dataset);names=sorted(training_names)\n    assert names and set(names)<=set(ds.meta['names'])\n    excluded=sorted(set(ds.meta['names'])-set(names))\n    dd=np.array([ds.dimensions[n] for n in names]);training_bounds=dict(min=dd.min(0).tolist(),max=dd.max(0).tolist())"),
        ('signature=dict(training_protocol=2,method=method,held_out=held_out,seed=seed,steps=steps,local_steps=local_steps,',
         'signature=dict(training_protocol="suzhou_original_v24_v1",method=method,held_out=held_out,trained_buildings=names,excluded_buildings=excluded,seed=seed,steps=steps,local_steps=local_steps,'),
        ("parameter_count=sum(p.numel() for p in model.parameters()),bounds=ds.meta['bounds'])",
         "parameter_count=sum(p.numel() for p in model.parameters()),bounds=training_bounds,excluded_buildings=excluded)"),
    ]
    for before, after in edits:
        if before not in s: raise ValueError(f'V24 source contract changed: {before[:70]}')
        s = s.replace(before, after)
    (ROOT/'train_v24.py').write_text(s, encoding='utf-8')
    src23 = WORK/'project_v2_3/fieldgen/height_transfer.py'
    s = src23.read_text(encoding='utf-8')
    edits = [
        ('from .', 'from fieldgen.'),
        ("names=plan['low'] if method=='source' else sorted(set(ds.meta['names'])-({held_out} if held_out else set()))",
         "names=sorted(plan['training_names'])\n    assert names and set(names)<=set(ds.meta['names'])\n    excluded=sorted(set(ds.meta['names'])-set(names))\n    dd=np.array([ds.meta['dims'][ds.meta['names'].index(n)] for n in names]);training_bounds=dict(min=dd.min(0).tolist(),max=dd.max(0).tolist())"),
        ("production=production,trained_buildings=names,bounds=ds.meta['bounds']", "production=production,trained_buildings=names,bounds=training_bounds,excluded_buildings=excluded"),
    ]
    for before, after in edits:
        if before not in s: raise ValueError(f'V23 source contract changed: {before[:70]}')
        s = s.replace(before, after)
    (ROOT/'train_v23.py').write_text(s, encoding='utf-8')


def prepare():
    ART.mkdir(parents=True, exist_ok=True)
    p = ART/'protocol.json'
    if p.exists():
        protocol = read(p)
        for path, sha in protocol['source_code_sha256'].items():
            if digest(WORK/path) != sha: raise ValueError(f'Frozen source changed: {path}')
        for path, sha in protocol['trainer_sha256'].items():
            if digest(ROOT/path) != sha: raise ValueError(f'Frozen trainer changed: {path}')
        return protocol
    a,b,e = read(OLD23/'manifest.json'),read(OLD24/'manifest.json'),read(EXTERNAL/'manifest.json')
    assert set(a['names']) == set(b['names']) and len(a['names']) == 28
    assert not set(a['names']) & set(e['names']) and len(e['names']) == 20
    assert a['dims'] == b['dims']
    dims = dict(zip(b['names'], b['dims']))
    for n,d in e['geometry'].items():
        assert not any(np.allclose(d['dims'], q, atol=.01, rtol=0) for q in a['dims'])
    groups=defaultdict(list)
    for n in sorted(dims): groups[family(dims[n])].append(n)
    high=sorted(g for g,ns in groups.items() if any(dims[n][2]>50 for n in ns))
    low=sorted(set(groups)-set(high));rng=np.random.default_rng(SEEDS[0]);rng.shuffle(high);rng.shuffle(low)
    buckets=[[] for _ in range(5)]
    for g in high+low:
        j=min(range(5),key=lambda j:(len(buckets[j]),sum(len(groups[x]) for x in buckets[j]),j))
        buckets[j].append(g)
    folds=[]
    for i,gg in enumerate(buckets):
        test=sorted(n for g in gg for n in groups[g]);train=sorted(set(dims)-set(test))
        assert {family(dims[n]) for n in train}.isdisjoint({family(dims[n]) for n in test})
        assert any(dims[n][2]>50 for n in train) and any(dims[n][2]<=50 for n in train)
        folds.append(dict(fold=i,train=train,test=test,test_families=gg))
    generate_trainers()
    protocol=dict(schema=1,created_local_date='2026-10-01',methods=METHODS,seeds=SEEDS,
        scope='Original V2.3 MLP and five included V2.4 configurations; complete-method comparison, not architecture-only causality.',
        training_buildings=sorted(dims),external_buildings=sorted(e['names']),folds=folds,
        recipes={'v23_mlp':dict(global_steps=4000,local_steps=300),
                 'v24_all':dict(global_steps=3000,local_steps=300)},
        production_seeds=SEEDS,development_seed=SEEDS[0],fixed_endpoints=True,
        external_used_for_training_or_selection=False,external_previously_viewed=True,
        external_label_quality=e['label_quality'],physical_validation='not_established',
        grid=dict(size=128,primary='x/L=-2..3,y/W=-1.5..1.5',historical='original per-building physical crop',
                  caveat='Geometry-scaled field of view differs from training crops; this transfer is reported explicitly.'),
        height_interval=[1.,1.5],strict_endpoints=True,height_band_edges=[1,1.02,1.10,1.25,1.5],
        height_weighting='Voronoi intervals on observed z/H, clipped to each requested band; spatial MSE then height-weighted RMSE per building; buildings equal weight.',
        metrics=['rmse_mps','mae_mps','r2','ssim','relative_rmse','top10_recall','top10_regret_mps'],
        regions=['full','roof','roof_core','roof_edge','windward','corner_flow_proxy','side','wake'],
        screening=dict(top_fraction=.1,absolute_speed_thresholds_mps=[4.,6.,8.],
                       height_target='lowest sampled height where true/predicted roof top-decile mean meets threshold',
                       thresholds_are_exploratory_not_engineering_standards=True),
        routing=dict(features=['H<=50','z/H band'],fit_on='original grouped out-of-fold results only',
                     criterion='equal-building roof RMSE',minimum_families=2,fallback='v23_mlp',
                     mean_ensemble='equal weight of all six included methods; same seed'),
        reliability=dict(roof_rmse_target_mps=.5,top10_regret_target_mps=.3,
                         targets='proposed study targets, not regulatory or established engineering thresholds',
                         minimum_buildings_per_stratum=5),
        dataset_manifest_sha256={str(x.relative_to(WORK)):digest(x) for x in [OLD23/'manifest.json',OLD24/'manifest.json',EXTERNAL/'manifest.json']},
        source_code_sha256={str(x.relative_to(WORK)):digest(x) for x in source_files()},
        trainer_sha256={x.name:digest(x) for x in [ROOT/'train_v23.py',ROOT/'train_v24.py']})
    write(p,protocol)
    write(ART/'jobs.json',build_jobs(protocol))
    return protocol


def build_jobs(p):
    jobs=[]
    # Compare V2.3 MLP with all five included V2.4 configurations.
    for method in METHODS:
        for seed in SEEDS:
            jobs.append(dict(kind='production',method=method,seed=seed,train=p['training_buildings'],test=p['external_buildings'],
                             key=f'production/{method}/seed_{seed}'))
        for fold in p['folds']:
            jobs.append(dict(kind='development',method=method,seed=SEEDS[0],train=fold['train'],test=fold['test'],
                             key=f"development/{method}/fold_{fold['fold']}"))
    return jobs


def train(job):
    p=prepare();out=ART/job['key'];out.mkdir(parents=True,exist_ok=True)
    names=sorted(job['train']);assert set(names)<=set(p['training_buildings'])
    assert set(names).isdisjoint(p['external_buildings'])
    if (out/'best.pt').exists():
        import torch
        c=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        assert c['trained_buildings']==names
        assert c.get('seed',c.get('signature',{}).get('seed'))==job['seed']
        if (out/'run.json').exists(): return
        raise ValueError('Incomplete checkpoint without run record')
    if job['method']=='v23_mlp':
        from train_v23 import fit
        from fieldgen.height_transfer import plan_for
        plan=plan_for(read(OLD23/'manifest.json'));plan['seed']=job['seed'];plan['training_names']=names
        plan['low']=[n for n in plan['low'] if n in names];plan['high']=[n for n in plan['high'] if n in names]
        write(out/'training_plan.json',plan)
        fit(OLD23,out,plan,'joint_balanced',production=job['kind']=='production')
    else:
        from train_v24 import fit
        fit(OLD24,out,job['method'],None,3000,300,job['seed'],names)


def run_queue():
    p=prepare();jobs=read(ART/'jobs.json')
    from bench.runner import process_alive
    lock=ART/'runner.lock'
    if lock.exists() and process_alive(read(lock)['pid']): raise RuntimeError('Study queue already active')
    write(lock,dict(pid=os.getpid(),started=time.time()))
    try:
        if not (ART/'audit.json').exists():
            from analysis_pipeline import audit
            audit(p)
        if not (ART/'historical/reproduction_audit.json').exists():
            from analysis_pipeline import reproduce_history
            reproduce_history()
        for i,j in enumerate(jobs):
            out=ART/j['key'];out.mkdir(parents=True,exist_ok=True)
            for action,done in [('train','run.json'),('evaluate','evaluation.json')]:
                if (out/done).exists():continue
                write(ART/'status.json',dict(state='RUNNING',pid=os.getpid(),job=j,index=i,total=len(jobs),action=action,updated=time.time()))
                with (out/f'{action}.log').open('a',encoding='utf-8') as f:
                    cmd=[sys.executable,str(ROOT/'study.py'),'worker','--index',str(i),'--action',action]
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    write(out/'process.json',dict(pid=child.pid,command=cmd));code=child.wait()
                if code:raise RuntimeError(f'{j["key"]} {action} exit {code}; see {out/action}.log')
            from analysis_pipeline import progress_report
            progress_report(p,jobs)
        from analysis_pipeline import finalize
        finalize(p,jobs)
        write(ART/'status.json',dict(state='MODEL_EXPERIMENTS_COMPLETE',pid=os.getpid(),updated=time.time(),
                                  physical_validation='CFD refinement and iteration sensitivity remain separate required research work'))
    except BaseException:
        write(ART/'status.json',dict(state='FAILED',error=traceback.format_exc(),updated=time.time()));raise
    finally:
        if lock.exists() and read(lock)['pid']==os.getpid():lock.unlink()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','audit','run','worker','report','finalize'])
    ap.add_argument('--index',type=int);ap.add_argument('--action',choices=['train','evaluate'])
    a=ap.parse_args();p=prepare()
    if a.command=='prepare': print(json.dumps(dict(methods=METHODS,jobs=len(read(ART/'jobs.json')),external=len(p['external_buildings']))));return
    from analysis_pipeline import audit,evaluate_job,progress_report,finalize
    jobs=read(ART/'jobs.json')
    if a.command=='audit':audit(p)
    elif a.command=='worker':
        j=jobs[a.index]
        if a.action=='train':train(j)
        else:evaluate_job(j,p)
    elif a.command=='run':run_queue()
    elif a.command=='report':progress_report(p,jobs)
    elif a.command=='finalize':finalize(p,jobs)


if __name__=='__main__':main()
