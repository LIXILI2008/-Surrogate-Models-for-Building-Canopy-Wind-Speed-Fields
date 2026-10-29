"""Generate a readable conclusion from completed frozen-protocol experiments."""
import numpy as np
from adaptation import OUT
from study import read

def main():
    p=read(OUT/'protocol.json');s=read(OUT/'summary.json');obs=read(OUT/'observations.json')
    assert read(OUT/'status.json')['state']=='EXPERIMENTS_COMPLETE'
    def result(kind,k,route):return next(r for r in s if (r['experiment'],r['k'],r['route'])==(kind,k,route))
    lines=['# 少样本适配完成结论','', '216个正式训练评价与18个策略候选训练已完成。原28栋预训练基线未覆盖。MLP完成适配实验；其余五种V2.4方法保留原统一比较，本轮未做其微调。','', '## 20栋整栋留出主分析','', '|支持栋数|不适配RMSE|适配RMSE|联合重训RMSE|仅支持集从头训练RMSE|适配负迁移栋数|','|---:|---:|---:|---:|---:|---:|']
    for k in p['k']:
        a=[result('main_grouped_cv',k,r) for r in ['zero_shot','adapt','joint','scratch']]
        lines.append(f"|{k}|"+'|'.join(f"{r['rmse']:.4f}" for r in a)+f"|{a[1]['negative_transfer_buildings']}/20|")
    z=result('main_grouped_cv',8,'zero_shot');a=result('main_grouped_cv',8,'adapt');j=result('main_grouped_cv',8,'joint')
    lines+=['',f"8栋适配的平均RMSE下降{100*(z['rmse']-a['rmse'])/z['rmse']:.1f}%，R²={a['r2']:.4f}，SSIM={a['ssim']:.4f}。联合重训RMSE={j['rmse']:.4f}，略低于适配，但不能仅凭均值差就认定稳定优势。",f"适配相对原模型的RMSE差值描述性95%区间：{a['delta_descriptive_ci95']} m/s。区间按建筑重采样，折间训练集重叠，不是严格独立样本推断。",'', '## 困难→容易诊断','', '|支持栋数|不适配RMSE|适配RMSE|联合重训RMSE|适配负迁移栋数|','|---:|---:|---:|---:|---:|']
    for k in p['k']:
        a=[result('retrospective_easy_query',k,r) for r in ['zero_shot','adapt','joint']]
        lines.append(f"|{k}|"+'|'.join(f"{r['rmse']:.4f}" for r in a)+f"|{a[1]['negative_transfer_buildings']}/6|")
    fold=next(f for f in p['folds'] if f['id']=='hard_to_easy')
    lines+=['','最终查询建筑：'+ '、'.join(p['buildings'][n]['label'] for n in fold['query'])+'。','内层验证选择全局参数微调；查询结果未用于重选策略。各支持数量的平均误差均高于原模型，不能宣称困难优先适配改善了容易建筑。潜在解释包括样本几何覆盖偏移与参数更新干扰已有规律；当前没有选择策略消融，不能把原因确定归于单一机制。','', '## 8栋适配仍失败的主分析案例','', '|建筑|实际W/L/H（m）|原RMSE|适配RMSE|误差增加|','|---|---|---:|---:|---:|']
    for n in p['buildings']:
        rr=[r for r in obs if r['experiment']=='main_grouped_cv' and r['k']==8 and r['route']=='adapt' and r['building']==n]
        before=float(np.mean([r['baseline_rmse'] for r in rr]));after=float(np.mean([r['rmse'] for r in rr]))
        if after>before:
            b=p['buildings'][n];dims='/'.join(f'{d:.3f}' for d in b['dims'])
            lines.append(f"|{b['label']}|{dims}|{before:.4f}|{after:.4f}|{after-before:.4f}|")
    lines+=['','## 可用于论文的结论与下一步','', '当前结果支持：少样本适配存在样本量依赖和几何依赖；原模型已有较强泛化，小样本更新可能造成负迁移；增加到8栋在当前回顾性主分析上改善均值，但未消除失败。不能宣称所有新几何均改善，不能宣称困难优先优于随机或几何多样性选样。','下一项有针对性的对照应固定相同查询建筑、K与训练预算，比较困难优先、随机、几何多样性三种支持集选择。保护原模型可比较源域回放或保守更新，但需仍由开发集选择，不能按最终查询结果重选。新实验应另立版本，不修改本轮协议和结果。','目前只完成MLP适配；若论文要比较不同架构的适配能力，还需为代表性V2.4模型补同划分适配。若只比较六模型跨几何基线与MLP的适配研究，应在文章标题、方法和结论明确范围。','20栋此前已被查看；“困难”由已有标签误差定义，所以是回顾性分析。源标签与新标签来源不同，部分700次CFD参考未达残差标准，未做网格独立性。可评价对这些固定参考标签的预测，不能把指标解释为已证实的真实城市风场精度。','', '数值单位：RMSE为m/s。少样本以建筑数计，每栋使用全部可用切片。适配1800步，从头/联合4300步，计算预算不同。']
    (OUT/'CONCLUSIONS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(OUT/'CONCLUSIONS.md')

if __name__=='__main__':main()
