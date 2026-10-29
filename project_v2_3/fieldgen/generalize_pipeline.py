"""Resumable detached CV/selection/refit pipeline; reserves never select a model."""
import argparse
import ctypes
import datetime
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
import torch
from .common import ROOT,read,write,digest
from .generalize import DEFAULT_DATA,make_plan,summaries,VARIANTS
from .train import Dataset


def cv_result(ds,root,variant,seeds,folds):
    pieces=[]
    for k in range(folds):
        values=[]
        for seed in seeds:
            with np.load(root/f'cv/{variant}_s{seed}_f{k}/oof.npz') as d:
                data={name:d[name].copy() for name in d.files}
            if values:
                assert np.array_equal(data['indices'],values[0]['indices'])
                assert np.array_equal(data['target'],values[0]['target'])
            values.append(data)
        data=values[0];data['pred']=np.mean([v['pred'] for v in values],axis=0);pieces.append(data)
    merged={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    return summaries(ds,merged['indices'],merged['target'],merged['pred'],merged['weight'])


def bundle(paths,output):
    ck=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    for c in ck[1:]:
        assert c['trained_buildings']==ck[0]['trained_buildings'] and c['bounds']==ck[0]['bounds']
        assert c['dataset_manifest_sha256']==ck[0]['dataset_manifest_sha256']
    result={k:v for k,v in ck[0].items() if k not in ('state','architecture')}
    result['members']=[dict(state=c['state'],architecture=c['architecture'],seed=c['seed']) for c in ck]
    result['member_checkpoint_sha256']=[digest(p) for p in paths]
    result['stage']='cv_selected_ensemble' if len(paths)>1 else 'cv_selected_single'
    Path(output).parent.mkdir(parents=True,exist_ok=True);torch.save(result,output)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default=str(DEFAULT_DATA));ap.add_argument('--output',default='artifacts/generalization');a=ap.parse_args()
    os.chdir(ROOT);data=Path(a.dataset).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    job=out/'job';job.mkdir(exist_ok=True)
    code_hashes={p.name:digest(p) for p in (ROOT/'fieldgen').glob('*.py')}
    identity=dict(code_hashes=code_hashes,dataset_manifest_sha256=digest(data/'manifest.json'))
    if (job/'identity.json').exists():
        if read(job/'identity.json')!=identity:raise ValueError('Code or data changed; use a fresh --output')
    else:write(job/'identity.json',identity)
    def status(stage,state='RUNNING',**kw):
        write(job/'status.json',dict(state=state,stage=stage,pid=os.getpid(),updated_at=datetime.datetime.now().astimezone().isoformat(),**kw))
    def run(stage,args,done):
        if Path(done).exists():return
        status(stage)
        with (job/(stage+'.log')).open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-u','-m',*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        ds=Dataset(data);plan=make_plan(ds.meta);write(out/'plan.json',plan)
        seed1,seed2=plan['seeds'];folds=len(plan['folds'])
        def train(variant,seed,k=None,refit=None,steps=4000,local=300):
            name=f'{variant}_s{seed}_f{k}' if refit is None else f'{variant}_s{seed}_{refit}'
            target=out/('cv' if refit is None else 'refits')/name
            args=['fieldgen.generalize','--dataset',data,'--output',target,'--variant',variant,'--seed',seed,
                  '--plan',out/'plan.json','--steps',steps,'--local-steps',local]
            args+=['--fold',k] if refit is None else ['--refit',refit]
            # Resume completed subjobs only. Partial checkpoints are preserved and not silently reused.
            run(name,args,target/'run.json')
            return target
        for variant in VARIANTS:
            for k in range(folds):train(variant,seed1,k)
        candidates={v:cv_result(ds,out,v,[seed1],folds) for v in VARIANTS}
        selected=min(candidates,key=lambda v:candidates[v]['macro_weighted_rmse'])
        write(out/'candidate_selection.json',dict(selected=selected,candidates=candidates,regression_used=False))
        for k in range(folds):train(selected,seed2,k)
        single=candidates[selected];ensemble=cv_result(ds,out,selected,[seed1,seed2],folds)
        # Require mean improvement and guard against a >5% worsening of worst-building error.
        use_ensemble=ensemble['macro_weighted_rmse']<single['macro_weighted_rmse'] and max(r['weighted_rmse'] for r in ensemble['buildings'])<=1.05*max(r['weighted_rmse'] for r in single['buildings'])
        seeds=[seed1,seed2] if use_ensemble else [seed1]
        chosen=ensemble if use_ensemble else single
        selection=dict(variant=selected,seeds=seeds,use_ensemble=use_ensemble,candidates=candidates,
            single=single,ensemble=ensemble,selected_cv=chosen,regression_used_for_selection=False,
            caveat='Development CV used for checkpoint and method selection; not an unbiased final test. Extremal anchors never scored. Previously viewed reserve is regression only.')
        write(out/'selection.json',selection)
        def durations(variant,seed):
            runs=[read(out/f'cv/{variant}_s{seed}_f{k}/run.json') for k in range(folds)]
            return int(np.median([r['global_step'] for r in runs])),int(np.median([r['local_step'] for r in runs]))
        dev=[];prod=[]
        for seed in seeds:
            steps,local=durations(selected,seed)
            dev.append(train(selected,seed,refit='development',steps=steps,local=local)/'best.pt')
            prod.append(train(selected,seed,refit='production',steps=steps,local=local)/'best.pt')
        bundle(dev,out/'development/best.pt');bundle(prod,out/'production/best.pt')
        # Matched baseline uses the same 25 development buildings and its own CV duration.
        bs,bl=durations('baseline',seed1)
        baseline=train('baseline',seed1,refit='development',steps=bs,local=bl)
        for name,path in [('baseline_regression',baseline),('selected_regression',out/'development')]:
            run(name,['fieldgen.evaluate','--dataset',data,'--run',path,'--output',out/name],out/name/'metrics.json')
        run('example',['fieldgen.predict','--checkpoint',out/'production/best.pt','--W','18','--L','16','--H','20','--z','24.37','--output',out/'example'],out/'example/W18_L16_H20_z24.37m.json')
        status('all_complete','COMPLETE',variant=selected,ensemble=use_ensemble)
    except BaseException as exc:
        status('failure','FAILED',error=str(exc));(job/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
