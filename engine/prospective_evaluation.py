"""Evaluate the preregistered six methods on all five held-out geometries."""
import argparse, os, time, shutil, subprocess, sys, traceback
from pathlib import Path
from study import ART, METHODS, SEEDS, read, write, digest
from validate_prospective_export import alive

OUT=ART/'prospective_five_v1'

def report():
    rows=[]
    for m in METHODS:
        rr=[read(OUT/'evaluation'/m/f'seed_{s}/evaluation.json') for s in SEEDS if (OUT/'evaluation'/m/f'seed_{s}/evaluation.json').exists()]
        if len(rr)!=len(SEEDS):continue
        vals={k:sum(r['groups']['all']['regions']['full'][k] for r in rr)/len(rr) for k in ['rmse_mps','mae_mps','r2']}
        vals['ssim']=sum(r['groups']['all']['ssim'] for r in rr)/len(rr)
        rows.append(dict(method=m,**vals))
    write(OUT/'evaluation/summary.json',dict(methods=rows,scope='Agreement with saved CFD; five buildings; three seeds; no adaptation on these labels'))
    lines=['# 新五栋：冻结六方法评价','','逐栋等权、高度间距加权，三种子均值。只评价H<z<1.5H；五栋不用于训练或调参。',
           '该结果测量相对保存CFD解的误差，不等于实验风场真实性验证。','',
           '|方法|RMSE m/s|MAE m/s|R2|SSIM|','|---|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+r['method']+'|'+'|'.join(f"{r[k]:.5f}" for k in ['rmse_mps','mae_mps','r2','ssim'])+'|')
    (OUT/'evaluation/REPORT.md').write_text('\n'.join(lines),encoding='utf8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int);a=ap.parse_args()
    frozen=read(OUT/'protocol.json');p=read(ART/'protocol.json');jobs=[]
    for method in METHODS:
        for seed in SEEDS:
            src=ART/'production'/method/f'seed_{seed}/best.pt'
            assert digest(src)==frozen['checkpoints_sha256'][f'{method}/{seed}']
            key=f'prospective_five_v1/evaluation/{method}/seed_{seed}';dest=ART/key;dest.mkdir(parents=True,exist_ok=True)
            if not (dest/'best.pt').exists():shutil.copy2(src,dest/'best.pt')
            assert digest(dest/'best.pt')==digest(src)
            jobs.append(dict(kind='production',method=method,seed=seed,key=key,train=p['training_buildings'],
                test=[s['directory'] for s in frozen['cases']],external_root=str(OUT/'export/dataset')))
    write(OUT/'evaluation/jobs.json',jobs)
    write(OUT/'evaluation/status.json',dict(state='WAITING_FOR_EXPORT',pid=os.getpid(),wait_pid=a.wait_pid,total=len(jobs)))
    if a.wait_pid:
        while alive(a.wait_pid):time.sleep(10)
    assert read(OUT/'export/status.json')['state']=='COMPLETE','Export incomplete'
    subprocess.run([sys.executable,str(Path(__file__).with_name('validate_prospective_export.py'))],check=True)
    audit=read(OUT/'export/dataset_validation.json');assert audit['complete_buildings']==5
    write(OUT/'evaluation/input_provenance.json',dict(protocol_sha256=digest(OUT/'protocol.json'),
        dataset_manifest_sha256=digest(OUT/'export/dataset/manifest.json'),evaluation_code_sha256=digest(Path(__file__).with_name('analysis_pipeline.py')),
        checkpoint_hashes=frozen['checkpoints_sha256'],no_training=True,adapted_models_included=False))
    from analysis_pipeline import evaluate_job
    for i,j in enumerate(jobs):
        if not (ART/j['key']/'evaluation.json').exists():evaluate_job(j,p)
        write(OUT/'evaluation/status.json',dict(state='RUNNING',pid=os.getpid(),completed=i+1,total=len(jobs),last=j['key']))
        print('EVALUATED',i+1,len(jobs),flush=True)
    report();write(OUT/'evaluation/status.json',dict(state='COMPLETE',completed=len(jobs),total=len(jobs)))

if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'evaluation/failure.json',dict(error=traceback.format_exc()));raise
