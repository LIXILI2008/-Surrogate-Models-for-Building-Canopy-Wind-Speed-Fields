"""Persistent sequential GPU queue with automatic evaluation and final report."""
import argparse
import datetime
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from .common import ROOT,read,write,digest
from .evaluate import average

METHODS=['unet_si','unet','fno','ufno','mlp_data','pinn']


def process_alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype=wintypes.BOOL
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:
            if ctypes.get_last_error()==5:raise PermissionError(f'Cannot inspect runner PID {pid}')
            return False
        try:
            code=wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
            return code.value==259 # STILL_ACTIVE; an openable process object may already be terminated.
        finally:kernel.CloseHandle(handle)
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False


def plan():
    meta=read(ROOT/'artifacts/dataset_128/manifest.json')
    d=dict(zip(meta['names'],meta['dims']))
    high=sorted(n for n,x in d.items() if x[2]>50)
    low=['bldg_W16_L20_H17','bldg_W8_L24_H12']
    cases=[]
    import numpy as np
    for name in low+high:
        rest=np.array([x for n,x in d.items() if n!=name]);x=np.array(d[name])
        inside=bool(((x>=rest.min(0)) & (x<=rest.max(0))).all())
        cases.append(dict(name=name,group='low_regression' if x[2]<=50 else 'high_in_range' if inside else 'high_extrapolation'))
    return dict(methods=METHODS,cases=cases,steps=3000,local_steps=300,seed=20260911,
        labels='Shared reversible relative-disturbance targets; unet_si is a label/loss package ablation, not a pure normalization ablation.',
        selection='Fixed training endpoints; no held-out early stopping. Same folds/data/supervised-point counts. Different parameter counts and PDE cost recorded.',
        inference='128x128 native benchmark resolution; no claim of 512x512 detail from upsampling.',
        limitations='One seed exploratory development comparison, low2/high4 plus1 extrapolation; not an independent publication-ready blind test.',
        physics='PINN interior steady 3D incompressible standard k-epsilon RANS, not laminar NS; no false inlet boundary on crop edges. Wall functions not reproduced; no claim of standalone CFD solver.',
        dataset_manifest_sha256=digest(ROOT/'artifacts/dataset_128/manifest.json'))


def report(out,p):
    summaries={};finished=0
    for method in ['v23_reference',*p['methods']]:
        per=[]
        for case in p['cases']:
            path=out/method/case['name']/'metrics.json'
            if path.exists():per.append(dict(case=case,metrics=read(path)));finished+=int(method!='v23_reference')
        groups={}
        for group in ('low_regression','high_in_range','high_extrapolation'):
            mm=[r['metrics'] for r in per if r['case']['group']==group]
            if not mm:continue
            groups[group]=dict(buildings=len(mm),rmse_mps=average([m['speed']['rmse'] for m in mm]),
                r2=average([m['speed']['r2'] for m in mm]),mean_slice_r2=average([m['speed']['mean_slice_r2'] for m in mm]),
                ssim=average([m['ssim_speed'] for m in mm]),disturbance_r2=average([m['disturbance']['r2'] for m in mm]),
                wake_rmse_mps=average([m['geometric_wake']['rmse'] for m in mm]),
                iou_at_4=average([m['iou_at_4']['iou'] for m in mm]),
                deficit_iou10=average([m['deficit_0.1']['iou'] for m in mm]),
                acceleration_iou10=average([m['acceleration_0.1']['iou'] for m in mm]),
                deficit_iou20=average([m['deficit_0.2']['iou'] for m in mm]),
                wake_deficit_iou10=average([m['wake_deficit_0.1']['iou'] for m in mm]))
            groups[group]['continuity_rms']=average([m.get('continuity_predicted',{}).get('rms') for m in mm])
            groups[group]['cfd_continuity_rms']=average([m.get('continuity_cfd_reference',{}).get('rms') for m in mm])
        summaries[method]=groups
    total=len(p['methods'])*len(p['cases'])
    write(out/'summary.json',dict(completed=finished,total=total,groups=summaries))
    lines=['# V2.4 标签与模型对比',f'已完成 {finished}/{total} 个训练和评估任务。',
        'V2.3实际是坐标MLP，不是U-Net。本轮新增真正U-Net/FNO/U-FNO及三维RANS PINN。',
        '标签 D=U/Uin(z)-1；真实m/s值未修改。IoU@4继续报告。减速/加速10%、20%区域用于区分背景与扰动。',
        '所有新模型共享高建筑的速度分量、压力、k、epsilon辅助监督；低建筑无此监督。mlp_data与pinn结构相同，只有后者加入物理损失。',
        'unet_si是旧尺度MSE与新标签/区域损失组合的消融，不将差异全部归因于归一化。',
        '同建筑留出、相同8192监督点/步、3000全局步+300局部步；参数量和运行时间不相同。固定终点、单种子开发性比较，不能宣称独立盲测。',
        '当前对比网格128×128；V2.3参考模型也在同网格、同标签上重评。原512网格旧分数不可直接混入。',
        '| 方法 | 分组 | 完成建筑 | RMSE m/s | R² | 逐图R² | SSIM | 扰动R² | 尾流RMSE | IoU@4 | 减速IoU10 | 加速IoU10 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method,groups in summaries.items():
        for group,m in groups.items():
            vals=[m[k] for k in ('rmse_mps','r2','mean_slice_r2','ssim','disturbance_r2','wake_rmse_mps','iou_at_4','deficit_iou10','acceleration_iou10')]
            lines.append('| '+method+' | '+group+f" | {m['buildings']} | "+' | '.join('—' if v is None else f'{v:.4f}' for v in vals)+' |')
    lines += ['','空并集不强设为1，记为未定义；逐切片JSON包含真实/预测正类占比、空类数量和误报率。',
              '高建筑采用已有接受的1200次/798次CFD标签，部分未达到全部原残差标准。',
              '三维RANS残差仅在高建筑训练几何内、离墙至少2m处约束。裁剪边界不是入口；本模型不复现Fluent近壁函数，不是无需数据的完整CFD求解器。']
    (out/'RESULTS.md').write_text('\n\n'.join(lines[:8])+'\n\n'+'\n'.join(lines[8:]),encoding='utf-8')
    return finished==total


def run_child(args,log):
    with log.open('a',encoding='utf-8') as f:
        child=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(log.with_suffix('.process.json'),dict(pid=child.pid,command=args))
        code=child.wait()
    if code:raise RuntimeError(f'Child exit {code}: {args}; see {log}')


def main():
    out=ROOT/'artifacts/comparison';out.mkdir(parents=True,exist_ok=True)
    lock=out/'runner.lock'
    # A live lock always blocks duplicate queues; dead PIDs can be resumed explicitly.
    if lock.exists():
        old=read(lock)
        if process_alive(old['pid']):raise RuntimeError(f"Runner already alive: {old['pid']}")
    write(lock,dict(pid=os.getpid()))
    try:
        p=plan()
        if (out/'plan.json').exists():assert read(out/'plan.json')==p
        else:write(out/'plan.json',p)
        for method in ['v23_reference',*p['methods']]:
            for case in p['cases']:
                name=case['name'];run=out/method/name;run.mkdir(parents=True,exist_ok=True)
                if (run/'metrics.json').exists():continue
                write(out/'status.json',dict(state='RUNNING',pid=os.getpid(),method=method,held_out=name,updated=datetime.datetime.now().isoformat()))
                if method!='v23_reference':
                    run_child(['-m','bench.train','--method',method,'--held-out',name,'--output',str(run),
                               '--steps',str(p['steps']),'--local-steps',str(p['local_steps'])],run/'train.log')
                args=['-m','bench.evaluate','--run',str(run),'--held-out',name]
                if method=='v23_reference':args+=['--legacy']
                run_child(args,run/'evaluate.log');report(out,p)
        assert report(out,p)
        # Code and fold weights/results; data cache/raw CFD intentionally excluded.
        dest=ROOT/'deliverables';dest.mkdir(exist_ok=True)
        archive=dest/'windfield_v2_4_code_results.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for path in ROOT.rglob('*'):
                if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts:continue
                rel=path.relative_to(ROOT)
                if rel.parts[0]=='deliverables' or 'preflight' in rel.parts or any(x.startswith('dataset_') for x in rel.parts):continue
                if path.name in ('resume.pt','runner.lock') or path.suffix=='.tmp':continue
                z.write(path,Path('project_v2_4')/rel)
        write(dest/'release.json',dict(archive=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,
            includes_training_cache=False,production_model_selected=False))
        write(out/'status.json',dict(state='COMPLETE',finished=datetime.datetime.now().isoformat(),archive=str(archive)))
    except BaseException:
        write(out/'status.json',dict(state='FAILED',error=traceback.format_exc(),time=datetime.datetime.now().isoformat()));raise
    finally:
        if lock.exists() and read(lock)['pid']==os.getpid():lock.unlink()


if __name__=='__main__':main()
