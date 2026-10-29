import argparse
import ctypes
import datetime
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
from .common import ROOT,read,write,digest
from .height_transfer import DEFAULT_DATA,plan_for,METHODS
from .train import Dataset,load
from .generalize import validate


def aggregate(out,plan,method):
    rows=[]
    for c in plan['cases']:
        r=read(out/f'folds/{method}/{c["held_out"]}/run.json')['validation']['buildings'][0]
        rows.append(dict(**r,in_training_dimension_bounds=c['in_training_dimension_bounds']))
    inside=[r for r in rows if r['in_training_dimension_bounds']]
    return dict(buildings=rows,in_range_macro_rmse=float(np.mean([r['rmse'] for r in inside])),
                in_range_macro_weighted_rmse=float(np.mean([r['weighted_rmse'] for r in inside])),
                all_five_macro_rmse=float(np.mean([r['rmse'] for r in rows])))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default=str(DEFAULT_DATA));ap.add_argument('--output',default='artifacts/height_transfer');a=ap.parse_args()
    os.chdir(ROOT);data=Path(a.dataset).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    job=out/'job';job.mkdir(exist_ok=True)
    identity=dict(code={p.name:digest(p) for p in (ROOT/'fieldgen').glob('*.py')},data=digest(data/'manifest.json'))
    if (job/'identity.json').exists():assert read(job/'identity.json')==identity,'Use a new output after code/data changes'
    else:write(job/'identity.json',identity)
    plan=plan_for(read(data/'manifest.json'));write(out/'plan.json',plan)
    def status(stage,state='RUNNING',**kw):
        write(job/'status.json',dict(stage=stage,state=state,pid=os.getpid(),updated_at=datetime.datetime.now().astimezone().isoformat(),**kw))
    def run(stage,args,done):
        if Path(done).exists():return
        status(stage)
        with (job/(stage+'.log')).open('w',encoding='utf8') as f:
            subprocess.run([sys.executable,'-u','-m',*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        common=['--dataset',data,'--plan',out/'plan.json']
        run('source',['fieldgen.height_transfer',*common,'--method','source','--output',out/'source'],out/'source/run.json')
        if not (out/'source_high_probe.json').exists():
            ds=Dataset(data);model,_=load(out/'source/best.pt','cpu')
            import torch
            torch.set_num_threads(4)
            write(out/'source_high_probe.json',dict(score=validate(model,ds,plan['high'],'cpu'),
                note='Low-only model queried beyond its training-height range; diagnostic, not an in-range generalization score.'))
        for method in METHODS:
            for c in plan['cases']:
                target=out/'folds'/method/c['held_out']
                args=['fieldgen.height_transfer',*common,'--method',method,'--held-out',c['held_out'],'--output',target]
                if method in ('transfer_replay','transfer_flexible'):args+=['--source',out/'source/best.pt']
                run(method+'_'+c['held_out'],args,target/'run.json')
        scores={m:aggregate(out,plan,m) for m in METHODS}
        selected=min(scores,key=lambda m:scores[m]['in_range_macro_weighted_rmse'])
        write(out/'selection.json',dict(selected=selected,candidates=scores,
            scope='Method development on four high-building in-range leave-one-out cases. No held-out early stopping. Not a fresh blind test.'))
        args=['fieldgen.height_transfer',*common,'--method',selected,'--production','--output',out/'production']
        if selected in ('transfer_replay','transfer_flexible'):args+=['--source',out/'source/best.pt']
        run('production',args,out/'production/run.json')
        for method in sorted(set(['joint_uniform',selected])):
            run('full_'+method,['fieldgen.height_evaluate','--dataset',data,'--experiment',out,'--method',method,'--output',out/('full_'+method)],out/('full_'+method)/'metrics.json')
        run('low_regression',['fieldgen.low_regression','--dataset',data,'--experiment',out],out/'low_regression/metrics.json')
        run('example',['fieldgen.predict','--checkpoint',out/'production/best.pt','--W','20','--L','12','--H','90','--z','99','--output',out/'example'],out/'example/W20_L12_H90_z99m.json')
        status('all_complete','COMPLETE',selected=selected)
    except BaseException as exc:
        status('failure','FAILED',error=str(exc));(job/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
