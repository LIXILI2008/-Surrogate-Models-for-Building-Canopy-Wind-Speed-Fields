"""Compare available continuation snapshots on joint supported CFD coordinates."""
from __future__ import annotations
from itertools import combinations
import numpy as np
from study import ART, EXTERNAL, read, write
from analysis_pipeline import grid, region_masks, scalar, screening, avg
from cfd_sensitivity import OUT, interpolate


def report():
    if not (OUT/'protocol.json').exists():return None
    protocol=read(OUT/'protocol.json');all_results=[]
    for entry in protocol['buildings']:
        folder=OUT/entry['label'];path=folder/'snapshots.json'
        snapshots=read(path) if path.exists() else []
        available=[s['additional_iterations'] for s in snapshots]
        records=[]
        for row in entry['planes'] if available else []:
            xy,_=grid(row['dims'])
            # Preserve the accepted primary cache and its support, not a rebuilt baseline.
            with np.load(EXTERNAL/row['cache']) as a:original=a['speed_mps'].copy();ov=a['valid'].copy()
            fields={0:(original,ov)}
            for extra in available:
                _,field,valid=interpolate(folder/f'delta_{extra}'/f"z{row['z']:g}.npz",row)
                fields[extra]=(field,valid)
            masks=region_masks(xy,row['dims']);uin=3*(row['z']/10)**.4
            for a,b in combinations(fields,2):
                ya,va=fields[a];yb,vb=fields[b];valid=va&vb
                records.append(dict(from_extra=a,to_extra=b,z_m=row['z'],ratio=row['z']/row['dims'][2],
                    regions={k:scalar(ya,yb,valid&m,uin) for k,m in masks.items()},
                    earlier_candidate_on_later_field=screening(yb,ya,valid&masks['roof']),
                    valid_fraction=float(valid.mean())))
        summaries=[]
        for a,b in combinations([0]+available,2):
            rows=[r for r in records if r['from_extra']==a and r['to_extra']==b]
            summaries.append(dict(from_extra=a,to_extra=b,planes=len(rows),
                roof_rmse_mps=avg([r['regions']['roof']['rmse_mps'] for r in rows]),
                wake_rmse_mps=avg([r['regions']['wake']['rmse_mps'] for r in rows]),
                candidate_regret_mps=avg([r['earlier_candidate_on_later_field']['top10_regret_mps'] for r in rows])))
        all_results.append(dict(label=entry['label'],building=entry['building'],available_extra_iterations=available,
            residual_passed_by_snapshot={str(s['additional_iterations']):s['residual_assessment']['all_passed'] for s in snapshots},
            complete=(folder/'result.json').exists(),comparisons=summaries,planes=records))
    result=dict(buildings=all_results,completed=sum(r['complete'] for r in all_results),total=len(all_results),
        aggregation='equal weight over the same five predefined source planes; not full-height aggregate',
        original_reference_replaced=False,mesh_sensitivity_complete=False,
        physical_validation='not established; no convergence threshold inferred from these observed changes')
    write(OUT/'summary.json',result)
    lines=['# CFD迭代敏感性：已有快照对照','',
           f"完整案例 {result['completed']}/{result['total']}；未完成案例的已有快照也列出。",
           '主评价原始标签保持不变。差异在共同有效坐标计算，每栋预先指定5层等权，不是全高度汇总。',
           '除与原始场比较外，还比较后续快照彼此的变化；残差达标与场变化分别记录。',
           '这些结果尚不能证明网格无关性或实际屋顶使用可靠性。','',
           '| 案例 | 追加迭代对照 | 层数 | 屋顶变化RMSE(m/s) | 尾流变化RMSE(m/s) | 原候选在后场的速度损失(m/s) |',
           '|---|---|---:|---:|---:|---:|']
    for r in all_results:
        if not r['comparisons']:lines.append(f"| {r['label']} | 待完成 | — | — | — | — |")
        for s in r['comparisons']:
            vals=['NA' if s[k] is None else f'{s[k]:.6f}' for k in ['roof_rmse_mps','wake_rmse_mps','candidate_regret_mps']]
            lines.append(f"| {r['label']} | +{s['from_extra']} → +{s['to_extra']} | {s['planes']} | "+' | '.join(vals)+' |')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    result=report()
    print('CFD summary:',result['completed'],'/',result['total'])
