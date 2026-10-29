"""Read-only evidence synthesis; no CFD solving and no new-model tuning."""
import json, re
from pathlib import Path
import numpy as np
from study import ART, WORK, read, write, digest

OUT=ART/'paper_evidence_v1'

def main():
    OUT.mkdir(exist_ok=True)
    obs=read(ART/'sampling_comparison_v1/observations.json');pairs=[];buildings=[]
    rng=np.random.default_rng(20261003)
    for kind in sorted({r['kind'] for r in obs}):
        for k in (1,2,4,8):
            maps={p:{(r['building'],r['seed']):r for r in obs if r['kind']==kind and r['k']==k and r['policy']==p} for p in ('hard','random','diverse')}
            assert set(maps['hard'])==set(maps['random'])==set(maps['diverse'])
            names=sorted({n for n,s in maps['hard']})
            for policy,m in maps.items():
                for name in names:
                    rr=[r for (n,s),r in m.items() if n==name];d=[float(x) for x in re.search(r'W([\d.]+)_L([\d.]+)_H([\d.]+)',name).groups()]
                    baseline=np.mean([r['baseline'] for r in rr]);rmse=np.mean([r['rmse'] for r in rr])
                    buildings.append(dict(kind=kind,k=k,policy=policy,building=name,W_over_H=d[0]/d[2],L_over_H=d[1]/d[2],
                        baseline=float(baseline),rmse=float(rmse),delta_rmse=float(rmse-baseline),negative_transfer=bool(rmse>baseline),
                        negative_seed_fraction=float(np.mean([r['rmse']>r['baseline'] for r in rr]))))
            for other in ('random','diverse'):
                delta=np.array([np.mean([r['rmse']-maps[other][key]['rmse'] for key,r in maps['hard'].items() if key[0]==n]) for n in names])
                samples=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
                pairs.append(dict(kind=kind,k=k,contrast='hard_minus_'+other,buildings=len(names),mean_delta=float(delta.mean()),
                    building_bootstrap_95_interval=np.quantile(samples,[.025,.975]).tolist(),hard_lower_error_buildings=int((delta<0).sum())))
    write(OUT/'paired_sampling.json',dict(pairs=pairs,building_rows=buildings,unit='m/s',
        interpretation='Retrospective; paired building bootstrap after averaging seeds; intervals descriptive, not multiplicity-adjusted; no z slices treated as independent'))
    lines=['# 选样策略配对分析','','同一建筑、同一种子、同一预算配对；先种子平均，再按建筑重采样5000次。区间为回顾性描述，不作多重比较校正后的显著性结论。','',
        '|评价|K|差值（困难优先减对照）|平均RMSE差|建筑bootstrap 95%区间|困难优先更低栋数|','|---|---:|---|---:|---|---:|']
    for r in pairs:
        lo,hi=r['building_bootstrap_95_interval'];lines.append(f"|{r['kind']}|{r['k']}|{r['contrast']}|{r['mean_delta']:.5f}|[{lo:.5f}, {hi:.5f}]|{r['hard_lower_error_buildings']}/{r['buildings']}|")
    (OUT/'选样配对分析.md').write_text('\n'.join(lines),encoding='utf8')
    # Inventory existing solver evidence; do not infer relative conservation from net flux alone.
    native=WORK/'artifacts/cfd_recovery';sources=[]
    for root,pattern in [(native/'convergence_audit','*/result.json'),(ART/'prospective_five_v1/export','P*/configuration_before.json'),
                         (ART/'cfd_sensitivity','summary.json'),(ART/'view_sensitivity','results.json')]:
        for p in root.glob(pattern):sources.append(dict(path=str(p),sha256=digest(p)))
    evidence=dict(sources=sources,continuation=read(ART/'cfd_sensitivity/summary.json'),
        existing_mass_reports=[str(p) for p in (native/'convergence_audit').glob('*/mass_flow.txt')],
        relative_mass_imbalance_established=False,mesh_sensitivity='cancelled_by_user_not_restarted',wind_tunnel_or_field_validation=False,
        label_source_confound='Original image-derived labels versus added native CFD; do not assign all performance differences to geometry',
        view_sensitivity=str(ART/'view_sensitivity/REPORT.md'),new_solver_iterations=0)
    write(OUT/'cfd_evidence_inventory.json',evidence)
    print('Evidence written',OUT,flush=True)

if __name__=='__main__':main()
