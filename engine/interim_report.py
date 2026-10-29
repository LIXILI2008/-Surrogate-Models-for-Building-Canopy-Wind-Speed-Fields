"""Read completed production evaluations without altering the active study queue."""
from __future__ import annotations
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import ART, METHODS, SEEDS, read, write
from analysis_pipeline import avg
from reporting import galleries, LABELS
from view_sensitivity import evaluate


def run():
    p=read(ART/'protocol.json')
    methods=[m for m in METHODS if all((ART/f'production/{m}/seed_{s}/evaluation.json').exists() for s in SEEDS)]
    out=ART/'interim_completed_models';out.mkdir(exist_ok=True)
    results={m:[read(ART/f'production/{m}/seed_{s}/evaluation.json') for s in SEEDS] for m in methods}
    pending=[m for m in METHODS if m not in methods]
    lines=['# 苏州20栋：已完成模型的阶段结果','',
           f'生成时间：{dt.datetime.now():%Y-%m-%d %H:%M:%S}。待完成模型：{", ".join(pending) or "无"}。',
           '每个已列模型均有3个生产种子；原28栋训练，20栋外部评价，919个严格屋顶上方切片。',
           '本页不是全模型最终结论。主评价为几何归一化公共视野；RMSE单位m/s，按高度间距加权、建筑等权。',
           '高组参考保持700次结果；尚未证明CFD网格无关性及全部案例的迭代稳定性。','',
           '| 方法 | 组别 | 屋顶RMSE | 全图RMSE | 全图R² | SSIM | 候选速度损失 |',
           '|---|---|---:|---:|---:|---:|---:|']
    scores={}
    for m,rr in results.items():
        scores[m]={}
        for group in ['low','high','all']:
            g=[r['groups'][group] for r in rr]
            v=dict(roof_rmse=avg([x['regions']['roof']['rmse_mps'] for x in g]),
                   full_rmse=avg([x['regions']['full']['rmse_mps'] for x in g]),
                   full_r2=avg([x['regions']['full']['r2'] for x in g]),
                   ssim=avg([x['ssim'] for x in g]),regret=avg([x['screening']['top10_regret_mps'] for x in g]))
            scores[m][group]=v
            lines.append('| '+m+' | '+group+' | '+' | '.join('NA' if x is None else f'{x:.4f}' for x in v.values())+' |')
    names=sorted(p['external_buildings'],key=lambda n:results[methods[0]][0]['buildings'][n]['label'])
    a=np.array([[avg([r['buildings'][n]['all_heights']['regions']['roof']['rmse_mps'] for r in results[m]]) for m in methods] for n in names])
    fig,ax=plt.subplots(figsize=(11,10),layout='constrained');im=ax.imshow(a,aspect='auto',cmap='magma_r')
    ax.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=20,ha='right')
    ax.set_yticks(range(len(names)),[results[methods[0]][0]['buildings'][n]['label'] for n in names])
    for i in range(len(names)):
        for j in range(len(methods)):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',color='white' if a[i,j]>a.max()/2 else 'black',fontsize=8)
    ax.set_title('Six-method comparison: roof RMSE, 3 seeds, equal building weights')
    fig.colorbar(im,ax=ax,label='m/s');fig.savefig(out/'roof_rmse.png',dpi=150);plt.close(fig)
    lines+=['','![20栋屋顶误差](roof_rmse.png)','',
            '[20栋、多高度速度/扰动/误差图集](gallery/index.html)','',
            '[相同坐标和有效域的视野敏感性对照](view_sensitivity/REPORT.md)','',
            '图集使用预定相对高度1.01、1.05、1.10、1.25、1.45附近的实际源层，图上注明实际z；图示固定种子20261001。',
            '视野对照固定权重、同一源层和共同有效坐标；它同时涉及图幅和有效分辨率变化，不能解释为纯架构消融。',
            '选择器将在所有模型的原始建筑折外结果齐全后拟合；本页不根据苏州20栋的误差调整训练或选择规则。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'status.json',dict(state='GENERATING',methods=methods,pending=pending))
    write(out/'summary.json',dict(methods=scores,pending=pending,final=False))
    galleries(p,methods=methods,out=out/'gallery_physical_scale',physical_scale=True)
    evaluate(p,methods=methods,out=out/'view_sensitivity')
    write(out/'status.json',dict(state='COMPLETE',methods=methods,pending=pending,study_complete=False))
    print('INTERIM REPORT COMPLETE',flush=True)


if __name__=='__main__':run()
