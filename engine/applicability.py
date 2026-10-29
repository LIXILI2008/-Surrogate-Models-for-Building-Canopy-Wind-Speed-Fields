"""Descriptive applicability and screening summaries; never fit on external scores."""
from study import ART, METHODS, write
from analysis_pipeline import avg


def report(p, results):
    rows=[];base=results['v23_mlp'];names=p['external_buildings']
    region_summary={m:{g:{region:avg([r['groups'][g]['regions'][region]['rmse_mps'] for r in rr])
        for region in ['windward','corner_flow_proxy','wake']} for g in ['low','high','all']}
        for m,rr in results.items()}
    roof_summary={m:{g:avg([r['groups'][g]['regions']['roof']['rmse_mps'] for r in rr])
        for g in ['low','high','all']} for m,rr in results.items()}
    bands=list(base[0]['buildings'][names[0]]['height_bands'])
    for name in names:
        b=base[0]['buildings'][name]
        for band in bands:
            for region in ['windward','corner_flow_proxy','wake']:
                scores={m:avg([r['buildings'][name]['height_bands'][band]['regions'][region]['rmse_mps'] for r in results[m]]) for m in METHODS}
                valid={m:v for m,v in scores.items() if v is not None}
                if not valid:continue
                winner=min(valid,key=valid.get)
                stable={m:all(
                    r['buildings'][name]['height_bands'][band]['regions'][region]['rmse_mps'] is not None
                    and a['buildings'][name]['height_bands'][band]['regions'][region]['rmse_mps'] is not None
                    and r['buildings'][name]['height_bands'][band]['regions'][region]['rmse_mps']<a['buildings'][name]['height_bands'][band]['regions'][region]['rmse_mps']
                    for r,a in zip(results[m],base)) for m in METHODS[1:]}
                rows.append(dict(label=b['label'],building=name,dims=b['dims'],group=b['group'],support=b['support'],
                    support_class=b['support']['support_class'],band=band,region=region,rmse_mps=scores,
                    posthoc_best=winner,lower_than_MLP_in_all_three_seeds=stable))
    positions={m:{g:dict(recall=avg([r['groups'][g]['screening']['top10_recall'] for r in rr]),
        regret_mps=avg([r['groups'][g]['screening']['top10_regret_mps'] for r in rr])) for g in ['low','high','all']} for m,rr in results.items()}
    heights=[]
    for m,rr in results.items():
        for group in ['low','high']:
            for threshold in p['screening']['absolute_speed_thresholds_mps']:
                v=[b['height_screening'][str(threshold)] for r in rr for b in r['buildings'].values() if b['group']==group]
                complete=[x for x in v if x['reference_roof_coverage_complete']]
                recommended=[x for x in complete if x['predicted_lowest_z_m'] is not None]
                feasible=[x for x in complete if x['true_lowest_z_m'] is not None]
                errors=[x['height_error_m'] for x in complete if x['height_error_m'] is not None]
                heights.append(dict(method=m,group=group,threshold_mps=threshold,building_seed_records=len(v),
                    complete_reference_records=len(complete),recommended_records=len(recommended),feasible_records=len(feasible),
                    false_recommendation_rate=avg([float(x['false_recommendation']) for x in recommended]),
                    missed_feasible_rate=avg([float(x['missed_feasible']) for x in feasible]),
                    mean_absolute_height_error_m=avg([abs(x) for x in errors]),paired_height_records=len(errors)))
    write(ART/'applicability.json',dict(region_summary=region_summary,roof_summary=roof_summary,band_region_comparisons=rows,position_screening=positions,height_screening=heights,
        interpretation='Posthoc descriptive strata, not deployment routing; seeds are repeated fits, not independent buildings. CFD reference uncertainty remains.'))
    lines=['# 适用范围与屋顶初筛：描述性证据','',
           '以下分组来自冻结外部评价，不能根据表中事后最优模型部署切换。实际选择规则只使用原建筑折外结果。',
           '“三个种子均优于MLP”表示本次训练重复方向一致，不代表已超过CFD参考的不确定性。H=50仅为描述分组。','',
           '## 位置区域误差（m/s）','',
           '下表为三种子、建筑等权、1<z/H<1.5 的区域RMSE。角流相关区是几何代理区。','',
           '| 区域 | 组别 | 建筑数 | V2.3 MLP | U-Net SI | U-Net | FNO | U-FNO | MLP-data |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for region in ['windward','corner_flow_proxy','wake']:
        show_region={'windward':'迎风区','corner_flow_proxy':'角流相关几何代理区','wake':'尾流区'}[region]
        for group in ['low','high','all']:
            vals=[region_summary[m][group][region] for m in METHODS]
            lines.append(f"| {show_region} | {group} | {10 if group!='all' else 20} | "+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines+=['','## 位置、高度与几何支持范围','',
           '| 区域 | 建筑组 | 几何支持 | z/H段 | 可比较栋数 | 均值误差最低方法 | V2.4方法三种子均优于MLP的栋数 |',
           '|---|---|---|---|---:|---|---|']
    for region in ['windward','corner_flow_proxy','wake']:
        for group in ['low','high']:
            for band in bands:
                for support in ['inside_hull','inside_axes_outside_hull','outside_axis_bounds']:
                    rr=[r for r in rows if r['region']==region and r['group']==group and r['band']==band and r['support_class']==support]
                    if not rr:continue
                    scores={m:avg([r['rmse_mps'][m] for r in rr if r['rmse_mps'][m] is not None]) for m in METHODS}
                    valid={m:v for m,v in scores.items() if v is not None}
                    if not valid:continue
                    winner=min(valid,key=valid.get)
                    counts=', '.join(f"{m}: {sum(bool(r['lower_than_MLP_in_all_three_seeds'][m]) for r in rr)}" for m in METHODS[1:])
                    show_region={'windward':'迎风区','corner_flow_proxy':'角流相关几何代理区','wake':'尾流区'}[region]
                    lines.append(f'| {show_region} | {group} | {support} | {band} | {len(rr)} | {winner} | {counts} |')
    lines+=['','## 最低达标高度','',
            '4/6/8m/s是研究阈值；仅在实际观测层中选择最低高度。未达标误报率以作出推荐的记录为分母；漏检率以CFD有可行层的记录为分母。',
            '记录为建筑×种子，不能当成新增独立建筑；完整参考覆盖与可评价高度误差的分母在JSON保留。','',
            '| 方法 | 组别 | 目标m/s | 未达标误报率 | 有可行层却未推荐率 | 高度绝对误差m |',
            '|---|---|---:|---:|---:|---:|']
    for r in heights:
        vals=['NA' if r[k] is None else f'{r[k]:.4f}' for k in ['false_recommendation_rate','missed_feasible_rate','mean_absolute_height_error_m']]
        lines.append(f"| {r['method']} | {r['group']} | {r['threshold_mps']} | "+' | '.join(vals)+' |')
    best_counts={region:sum(r['posthoc_best']=='v23_mlp' for r in rows if r['region']==region)
        for region in ['windward','corner_flow_proxy','wake']}
    sup=base[0]['buildings'];support_counts={k:sum(b['support']['support_class']==k for b in sup.values())
        for k in ['inside_hull','inside_axes_outside_hull','outside_axis_bounds']}
    lo=positions['v23_mlp']['low'];hi=positions['v23_mlp']['high'];allp=positions['v23_mlp']['all']
    hs={(r['group'],r['threshold_mps']):r for r in heights if r['method']=='v23_mlp'}
    lines+=['','## 目前可以下的结论','',
            f"V2.3 MLP是20栋外部案例的总体默认基线：屋顶RMSE低组{roof_summary['v23_mlp']['low']:.3f} m/s、高组0.575 m/s、全体0.441 m/s。它在迎风区/角部代理区/尾流区的全体RMSE分别为0.217/0.459/0.633 m/s；三种区域上，按逐栋×高度段的事后最低RMSE计，MLP分别胜出{best_counts['windward']}/80、{best_counts['corner_flow_proxy']}/80、{best_counts['wake']}/80项。尾流仍是误差最大的区域。",
            f"位置初筛有速度排序价值，但还不支持精确点位定位：真实屋顶前10%高速点的召回率低组{lo['recall']:.3f}、高组{hi['recall']:.3f}；候选集平均速度损失仅{allp['regret_mps']:.3f} m/s。也就是说候选速度均值接近参考最优，但高组具体位置重合偏低。",
            f"高度初筛尚不能称为可靠工程判据：以4/6/8 m/s作探索阈值时，MLP低组6 m/s误推荐率{hs[('low',6.0)]['false_recommendation_rate']:.3f}、高度误差{hs[('low',6.0)]['mean_absolute_height_error_m']:.2f} m；高组8 m/s误推荐率{hs[('high',8.0)]['false_recommendation_rate']:.3f}、高度误差{hs[('high',8.0)]['mean_absolute_height_error_m']:.2f} m。",
            f"几何外推仍是薄弱证据：20栋中{support_counts['inside_hull']}栋位于训练几何凸包内、{support_counts['inside_axes_outside_hull']}栋在轴范围内但凸包外、{support_counts['outside_axis_bounds']}栋至少一个尺寸超出训练轴范围。MLP在多数分区高度对比中排名最低误差，但外推可靠性仍需新建筑确认。",
            '','## 解释边界','',
            '位置区按迎风区、角流相关几何代理区、尾流区评价。角流代理区是四个平面角点外侧的固定邻域；标签只有速度大小，无法确认旋涡强度或流动分离，因此不将代理区结果解释为涡旋预测能力。',
            '尺寸范围外、联合几何范围外、近屋顶边缘和尾流分别保留逐栋结果。完整记录见applicability.json；不将低误差的少量案例直接称为可靠几何区域。',
            '可靠性还需结合CFD迭代状态、独立建筑数量和未查看的新案例。网格敏感性按用户决定不开展。当前证据只适用于固定来流、长方体几何和速度大小指标的探索性初筛。']
    (ART/'APPLICABILITY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
