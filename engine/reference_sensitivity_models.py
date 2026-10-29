"""Fixed predictions scored against existing CFD continuation snapshots only."""
from pathlib import Path
import numpy as np
from study import ART,EXTERNAL,METHODS,SEEDS,read,write
from cfd_sensitivity import OUT as CFD,interpolate
from analysis_pipeline import grid,region_masks,scalar

def main():
    out=ART/'paper_evidence_v1/reference_sensitivity';out.mkdir(parents=True,exist_ok=True)
    write(out/'status.json',dict(state='RUNNING'))
    protocol=read(CFD/'protocol.json');results=[]
    for label,extra in [('L03',500),('U06',100)]:
        entry=next(e for e in protocol['buildings'] if e['label']==label);planes=[]
        for row in entry['planes']:
            xy,_=grid(row['dims']);_,later,lv=interpolate(CFD/label/f'delta_{extra}'/f"z{row['z']:g}.npz",row)
            with np.load(EXTERNAL/row['cache']) as a:old=a['speed_mps'].copy();valid=a['valid']&lv
            planes.append((row,old,later,valid,region_masks(xy,row['dims'])))
        for method in METHODS:
            for seed in SEEDS:
                folder=ART/'production'/method/f'seed_{seed}';meta=read(folder/'prediction_rows.json');pred=np.load(folder/'predictions.npy',mmap_mode='r')
                for row,old,later,valid,masks in planes:
                    indices=[i for i,r in enumerate(meta) if r['building']==row['building'] and abs(r['z_m']-row['z'])<1e-6]
                    if len(indices)!=1:raise ValueError('No unique original prediction for frozen CFD plane')
                    v=np.asarray(pred[indices[0]]);uin=3*(row['z']/10)**.4
                    metrics={}
                    for region in ('full','roof','wake'):
                        mask=valid&masks[region]
                        metrics[region]=dict(original=scalar(old,v,mask,uin),continued=scalar(later,v,mask,uin),reference_change=scalar(old,later,mask,uin))
                    results.append(dict(label=label,method=method,seed=seed,z=row['z'],additional_iterations=extra,regions=metrics))
    write(out/'results.json',dict(rows=results,scope='Two existing CFD cases, five predefined planes; joint support; unchanged original predictions; no new solver iterations'))
    lines=['# 已有CFD续算对模型评价的影响','','固定原模型预测，原参考与续算参考使用共同有效点。每例5层等权后对3种子平均；不是全高度或全建筑结果。','',
        '|案例|方法|原参考整场RMSE|续算参考整场RMSE|原参考尾流RMSE|续算参考尾流RMSE|','|---|---|---:|---:|---:|---:|']
    for label in ('L03','U06'):
        for method in METHODS:
            rows=[r for r in results if r['label']==label and r['method']==method]
            vals=[np.mean([r['regions'][reg][ref]['rmse_mps'] for r in rows]) for reg,ref in [('full','original'),('full','continued'),('wake','original'),('wake','continued')]]
            lines.append('|'+label+'|'+method+'|'+'|'.join(f'{v:.6f}' for v in vals)+'|')
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8');write(out/'status.json',dict(state='COMPLETE',rows=len(results)))

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write(ART/'paper_evidence_v1/reference_sensitivity/failure.json',dict(error=traceback.format_exc()));raise
