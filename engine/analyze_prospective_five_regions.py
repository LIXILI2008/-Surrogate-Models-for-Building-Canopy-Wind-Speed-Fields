"""Read-only frozen-prediction audit; writes only a fresh analysis directory."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_pipeline import Labels, region_masks, scalar, aggregate_slices
from study import METHODS, SEEDS

ROOT = Path(__file__).resolve().parent
SRC = ROOT/'artifacts/prospective_five_v1'
REGIONS = ['full', 'roof', 'windward', 'side', 'wake']
METRICS = ['rmse_mps', 'mae_mps', 'r2', 'relative_rmse', 'mean_slice_r2']
CN = dict(full='全域', roof='屋顶', windward='迎风', side='侧缘', wake='尾流')
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out.resolve();assert not out.exists(), 'Use a NEW output directory'
    assert not out.is_relative_to(SRC), 'Keep analysis outside frozen source'
    out.mkdir(parents=True)
    # Hash every frozen artifact before and after; no source writes or inference.
    before={p.relative_to(SRC).as_posix():sha(p) for p in SRC.rglob('*') if p.is_file()}
    protocol=read(SRC/'protocol.json'); prov=read(SRC/'evaluation/input_provenance.json')
    assert before['protocol.json']==prov['protocol_sha256']
    assert before['export/dataset/manifest.json']==prov['dataset_manifest_sha256']
    assert sha(ROOT/'analysis_pipeline.py')==prov['evaluation_code_sha256']
    assert read(SRC/'evaluation/status.json')['state']=='COMPLETE'
    cases=protocol['cases'];ds=Labels('production',[c['directory'] for c in cases],SRC/'export/dataset')
    labels=[];coverage=[]
    for row in ds.rows:
        assert sha(ds.external_root/row['cache'])==row['cache_sha256']
        assert sha(Path(row['source']))==row['source_sha256']
        xy,bounds,y,valid,dist=ds.get(row)
        assert y.shape==valid.shape==(128,128) and np.isfinite(y[valid]).all()
        masks=region_masks(xy,row['dims']);labels.append((y,valid,masks))
        coverage.append(dict(building=row['building'],z_m=row['z'],regions={r:dict(geometric_pixels=int(masks[r].sum()),valid_pixels=int((masks[r]&valid).sum())) for r in REGIONS}))
    for c in cases:
        rr=[r for r in ds.rows if r['building']==c['directory']]
        assert np.allclose([r['z'] for r in rr],c['evaluation_z_m'])
        assert all(r['dims']==[c['W'],c['L'],c['H']] for r in rr)
    detail=[];macro=[];deltas=[]
    for method in METHODS:
        for seed in SEEDS:
            folder=SRC/'evaluation'/method/f'seed_{seed}';ev=read(folder/'evaluation.json')
            assert sha(folder/'best.pt')==protocol['checkpoints_sha256'][f'{method}/{seed}']==ev['checkpoint_sha256']
            assert sha(folder/'predictions.npy')==ev['prediction_sha256']
            assert ev['evaluation_code_sha256']==prov['evaluation_code_sha256']
            mapping=read(folder/'prediction_rows.json');saved=read(folder/'per_slice.json')
            pred=np.load(folder/'predictions.npy',mmap_mode='r');assert pred.shape==(len(ds.rows),128,128)
            assert len(mapping)==len(saved)==len(ds.rows)
            recalculated=[]
            for i,(row,(y,valid,masks)) in enumerate(zip(ds.rows,labels)):
                assert mapping[i]==dict(building=row['building'],z_m=row['z'],source_index=row['source_index'])
                assert saved[i]['building']==row['building'] and saved[i]['z_m']==row['z']
                assert np.isfinite(pred[i]).all()
                rr=dict(saved[i]);rr['regions']={r:scalar(y,pred[i],masks[r]&valid,3*(row['z']/10)**.4) for r in REGIONS}
                for r in REGIONS:
                    for k,v in rr['regions'][r].items():
                        old=saved[i]['regions'][r][k]
                        assert (v is None and old is None) or np.isclose(v,old,rtol=1e-10,atol=1e-10)
                recalculated.append(rr)
            for c in cases:
                name=c['directory'];agg=aggregate_slices([r for r in recalculated if r['building']==name])
                for region in REGIONS:
                    vals=agg['regions'][region];old=ev['buildings'][name]['all_heights']['regions'][region]
                    for k in METRICS:
                        assert vals[k] is not None and np.isfinite(vals[k])
                        deltas.append(abs(vals[k]-old[k]));assert np.isclose(vals[k],old[k],rtol=1e-10,atol=1e-10)
                    detail.append(dict(case=c['id'],building=name,method=method,seed=seed,region=region,slices=agg['slices'],**vals))
            for region in REGIONS:
                vals={k:float(np.mean([r[k] for r in detail if r['method']==method and r['seed']==seed and r['region']==region])) for k in METRICS}
                for k in METRICS: assert np.isclose(vals[k],ev['groups']['all']['regions'][region][k],rtol=1e-10,atol=1e-10)
                macro.append(dict(case='MACRO',method=method,seed=seed,region=region,buildings=5,**vals))
            print('verified',method,seed,flush=True)
    summary=[]
    for case in [c['id'] for c in cases]+['MACRO']:
        for region in REGIONS:
            for method in METHODS:
                rr=[r for r in detail+macro if r['case']==case and r['region']==region and r['method']==method]
                assert len(rr)==3
                summary.append(dict(case=case,region=region,method=method,seed_count=3,metrics={k:dict(mean=float(np.mean([r[k] for r in rr])),sd=float(np.std([r[k] for r in rr],ddof=1)),min=min(r[k] for r in rr),max=max(r[k] for r in rr)) for k in METRICS}))
    write(out/'building_region_seed_metrics.json',detail);write(out/'macro_region_seed_metrics.json',macro)
    write(out/'three_seed_summary.json',summary);write(out/'region_coverage.json',coverage)
    def get(case,region,method,k):return next(r['metrics'][k] for r in summary if r['case']==case and r['region']==region and r['method']==method)
    for metric in ['rmse_mps','mae_mps','r2']:
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        allvalues=[r['metrics'][metric]['mean'] for r in summary];vmin=min(allvalues) if metric=='r2' else 0;vmax=max(allvalues)
        for ax,case in zip(axes.flat,[c['id'] for c in cases]+['MACRO']):
            matrix=np.array([[get(case,r,m,metric)['mean'] for r in REGIONS] for m in METHODS])
            im=ax.imshow(matrix,aspect='auto',cmap='viridis' if metric=='r2' else 'magma_r',vmin=vmin,vmax=vmax)
            ax.set(xticks=range(5),xticklabels=REGIONS,yticks=range(6),yticklabels=METHODS,title=case)
            for i in range(6):
                for j in range(5):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if ((matrix[i,j]-vmin)/(vmax-vmin)<.48 if metric=='r2' else (matrix[i,j]-vmin)/(vmax-vmin)>.48) else 'black')
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label=metric)
        fig.suptitle(f'{metric}: three-seed mean; MACRO = equal-building mean; saved CFD agreement')
        fig.savefig(out/f'{metric}_heatmaps.png',dpi=170);fig.savefig(out/f'{metric}_heatmaps.svg');plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(19,5),layout='constrained')
    for ax,region in zip(axes,REGIONS):
        for i,m in enumerate(METHODS):
            vals=[r['rmse_mps'] for r in macro if r['region']==region and r['method']==m];st=get('MACRO',region,m,'rmse_mps')
            ax.errorbar(i,st['mean'],yerr=st['sd'],fmt='o',color='black',capsize=4)
            ax.scatter(np.array([i-.13,i,i+.13]),vals,c=['#1b9e77','#d95f02','#7570b3'],s=22,zorder=3)
        ax.set(title=region,xticks=range(6),xticklabels=METHODS,ylabel='Building-macro RMSE (m/s)',ylim=(0,None));ax.tick_params(axis='x',rotation=60);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Equal-building macro: mean ± sample SD across 3 seeds; colored points = seeds 20261001/02/03')
    fig.savefig(out/'macro_seed_spread.png',dpi=170);fig.savefig(out/'macro_seed_spread.svg');plt.close(fig)
    lines=['# 新五栋冻结评价：逐建筑、逐区域、六方法 × 三种子','',
    '相对保存 CFD 解的误差不等于实验风场真实性验证。导出审计 convergence_certified=false；仅衡量模型对保存数值解的复现。五栋不用于训练、调参、修改掩膜或模型选择。','',
    '## 协议与复核','',
    '18 组冻结检查点与预测哈希均匹配；评价协议、标签清单、评价代码哈希匹配。301 个高度的数值标签缓存及其源 NPZ 哈希已核对；每组预测行顺序、建筑、高度、尺寸、128×128 网格一致。复算所有区域逐切片指标、逐栋高度汇总和跨栋宏平均，与原评价逐项一致（容差 1e-10）。未执行训练或重新推理。','',
    '风向 +x，L 沿 x、W 沿 y。采用冻结名义 W/L/H；不根据 measured_WLH 重新调整几何或掩膜。裁剪域 x∈[-2L,3L]、y∈[-1.5W,1.5W]。严格 H<z<1.5H。数值标签为 speed_mps 标量风速，不含风向/涡结构验证。','',
    '|建筑|W/L/H (m)|切片数|协议中的几何支持说明|','|---|---|---:|---|']
    for c in cases: lines.append(f"|{c['id']}|{c['W']}/{c['L']}/{c['H']}|{len(c['evaluation_z_m'])}|{c['reason']}|")
    lines+=['','## 区域与指标口径','',
    '所有区域与原 valid 掩膜及裁剪域取交集。屋顶为水平投影上方全高度区域，并非仅屋面或首米。迎风/尾流等区域可能被裁剪域截断，未扩大掩膜；区域不是全域的穷尽分区。','',
    '|区域|冻结定义|','|---|---|','|屋顶 roof|abs(x)≤L/2，abs(y)≤W/2|','|迎风 windward|−L/2−H≤x<−L/2，abs(y)≤W/2|','|侧缘 side|abs(x)≤L/2，W/2<abs(y)≤W/2+min(H,W)|','|尾流 wake|L/2<x≤L/2+2H，abs(y)≤max(W,H/4)|','',
    '每层有效像素等权；按 z/H 相邻中点构造高度权重，端点延伸至 1 和 1.5，归一化后得到 w。MSE=Σw·层MSE，RMSE=√MSE，MAE=Σw·层MAE。R²=1−MSE/[Σw·E(y²)−(Σw·E(y))²]；不同于逐层 R² 的平均，后者另存 mean_slice_r2。relative_rmse=√Σw·层MSE/U(z)²，U(z)=3(z/10)^0.4。负 R² 原样保留。','',
    '跨栋宏平均：每种子先对五栋对应指标算术平均（不是合并像素 RMSE，也不是先平均 MSE 再开方）；之后给出三种子的均值±样本标准差（ddof=1）。标准差是种子波动，不是置信区间或五栋总体不确定度。切片不作为独立建筑样本。SSIM 原评价仅为全域指标，本报告不构造区域 SSIM。','',
    '## 六方法逐建筑及宏平均','', '每格为三种子均值 ± 样本标准差。完整单种子值见后续明细与 JSON。']
    for case in [c['id'] for c in cases]+['MACRO']:
        lines+=['',f'### {case}','','|区域|方法|RMSE m/s|MAE m/s|R²|相对 RMSE|','|---|---|---:|---:|---:|---:|']
        for region in REGIONS:
            for m in METHODS:
                cells=[f"{get(case,region,m,k)['mean']:.5f} ± {get(case,region,m,k)['sd']:.5f}" for k in METRICS[:4]]
                lines.append('|'+ '|'.join([CN[region],m,*cells])+'|')
    lines+=['','## 可视化','','![RMSE](rmse_mps_heatmaps.png)','','![MAE](mae_mps_heatmaps.png)','','![R2](r2_heatmaps.png)','','![种子波动](macro_seed_spread.png)','',
    '## 单种子完整明细','','|建筑|区域|方法|种子|RMSE m/s|MAE m/s|R²|相对 RMSE|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in detail+macro:lines.append('|'+ '|'.join([r['case'],CN[r['region']],r['method'],str(r['seed']),*[f'{r[k]:.7f}' for k in METRICS[:4]]])+'|')
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    after={p.relative_to(SRC).as_posix():sha(p) for p in SRC.rglob('*') if p.is_file()};assert before==after
    write(out/'audit.json',dict(state='PASS',frozen_source_unchanged=True,source_files_sha256=before,analysis_script_sha256=sha(Path(__file__)),runs=18,slices_per_run=len(ds.rows),building_region_seed_rows=len(detail),macro_region_seed_rows=len(macro),max_recomputed_aggregate_abs_difference=max(deltas),convergence_certified=False,no_training=True,no_inference=True))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()


