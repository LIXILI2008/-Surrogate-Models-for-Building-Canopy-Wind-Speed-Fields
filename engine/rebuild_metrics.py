"""Recompute evaluation JSONs from saved predictions after region definitions change."""
from __future__ import annotations
import time
import numpy as np
from study import ART, METHODS, SEEDS, read, write, digest
from analysis_pipeline import Labels, slice_metrics, summarize


def main():
    p=read(ART/'protocol.json');jobs=read(ART/'jobs.json')
    for ji,job in enumerate(jobs,1):
        out=ART/job['key'];ds=Labels(job['kind'],job['test'])
        pred=np.load(out/'predictions.npy',mmap_mode='r')
        if pred.shape!=(len(ds.rows),128,128):
            raise ValueError(f"prediction shape mismatch for {job['key']}: {pred.shape}")
        rows=[];baseline=[];started=time.time()
        for i,row in enumerate(ds.rows):
            xy,bounds,y,valid,dist=ds.get(row)
            rows.append(slice_metrics(row,y,pred[i],valid,xy,dist))
            if job['method']=='v23_mlp' and job['seed']==SEEDS[0]:
                baseline.append(slice_metrics(row,y,np.full_like(y,3*(row['z']/10)**.4),valid,xy,dist))
        result=summarize(rows,job,p);result.update(prediction_sha256=digest(out/'predictions.npy'),
            evaluation_code_sha256=digest(__file__),metrics_rebuilt_for_regions=True)
        write(out/'per_slice.json',rows);write(out/'evaluation.json',result)
        if baseline:
            write(out/'baseline_per_slice.json',baseline)
            write(out/'baseline_evaluation.json',summarize(baseline,dict(job,method='inlet_baseline'),p))
        print(f"[{ji}/{len(jobs)}] {job['key']} {len(rows)} slices",flush=True)
    from analysis_pipeline import progress_report
    progress_report(p,jobs)
    write(ART/'status.json',dict(state='METRICS_REBUILT',methods=METHODS,jobs=len(jobs),updated=time.time()))


if __name__=='__main__':main()
