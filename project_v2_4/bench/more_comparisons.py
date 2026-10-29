"""Generate additional existing-label holdout panels, without training."""
import html
from zipfile import ZipFile, ZIP_DEFLATED
from . import explain_versions as existing
from .common import ROOT, read, write


def main():
    out=ROOT/'deliverables/height_comparisons'
    out.mkdir(parents=True,exist_ok=True)
    names=['bldg_W16_L20_H17','bldg_W8_L24_H12','bldg_W11_L12_H70',
           'bldg_W30_L16_H80','bldg_W24_L8_H100','bldg_W30_L16_H100']
    records=read(ROOT/'artifacts/dataset_128/manifest.json')['records']
    plans=[]; selection=[]
    # Select by geometry and height alone, before examining model errors.
    for name in names:
        candidates=[r for r in records if r['building']==name]
        used=set()
        for ratio in [1.05,1.20,1.35,1.50]:
            row=min((r for r in candidates if r['z'] not in used),key=lambda r:(abs(r['ratio']-ratio),r['z']))
            z=row['z'];used.add(z)
            slug=name.removeprefix('bldg_')+'_z'+f'{z:g}'.replace('.','p')
            plans.append((name,z,slug))
            selection.append(dict(building=name,z=z,ratio=row['ratio'],target_ratio=ratio,figure=slug+'.png'))
    write(out/'selection.json',selection)
    print('Selected '+str(len(plans))+' panels by height only',flush=True)
    existing.OUT=out
    rows=existing.examples(plans)
    sections=[]; md=['# 低高建筑多高度对比图','',
        '共24张新增对比图：2栋低建筑、4栋范围内高建筑，每栋4个高度。按目标z/H=1.05、1.20、1.35、1.50选择最近的可用参考切片，实际高度见下表；不按模型分数筛图。',
        '每图沿用原3行6列格式。列：参考标签、V2.3、U-Net、FNO、U-FNO、PINN；行：速度、相对扰动、绝对误差。',
        '所有预测使用排除目标建筑全部高度的既有权重；128×128共同评价网格。每次重推RMSE与存档逐图指标差异须小于1e-5 m/s。',
        '低建筑色标0–6、高建筑0–13 m/s；扰动±0.5；绝对误差低0–0.5、高0–1.5 m/s。超范围颜色饱和，但指标用未截断数值。白色为标签不可用区域，虚线为几何投影。',
        '下表“最低RMSE模型”是查看参考标签后的描述性结果，不能直接作为新建筑的自动选模规则。低建筑参考标签为校准图像，高建筑为接受的Fluent数值标签。','',
        '|建筑 W×L×H m|实际z m|z/H|最低RMSE模型|RMSE m/s|图像|',
        '|---|---:|---:|---|---:|---|']
    for row in rows:
        dims=row['dims'];z=row['z'];ratio=z/dims[2]
        label=f'W{dims[0]:g} × L{dims[1]:g} × H{dims[2]:g} m · z={z:g} m · z/H={ratio:.3f}'
        metrics=row['single_slice'];winner=min(metrics,key=lambda m:metrics[m]['speed']['rmse'])
        best=metrics[winner]['speed']['rmse']
        cells=[]
        for method in existing.METHODS:
            s=metrics[method]
            cells.append('<tr><td>'+html.escape(method)+'</td>'+''.join('<td>'+existing.fmt(v)+'</td>' for v in [s['speed']['rmse'],s['speed']['r2'],s['ssim_speed'],s['deficit_0.1']['iou'],s['geometric_wake']['rmse']])+'</tr>')
        sections.append(f'<section><h2>{label}</h2><a href="{row["figure"]}" target="_blank"><img loading="lazy" src="{row["figure"]}" alt="{label} 六模型同网格比较"></a><details><summary>查看该图指标</summary><table><tr><th>模型</th><th>RMSE m/s</th><th>R²</th><th>SSIM</th><th>减速IoU10%</th><th>尾流RMSE m/s</th></tr>'+''.join(cells)+'</table></details></section>')
        md.append(f'|{dims[0]:g}×{dims[1]:g}×{dims[2]:g}|{z:g}|{ratio:.3f}|{winner}|{best:.4f}|[打开]({row["figure"]})|')
    (out/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    intro='24张新增对比图 · 6栋建筑各4个高度 · 同128×128网格 · 整建筑留出预测'
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>低高建筑多高度对比</title><style>body{font-family:Arial,"Microsoft YaHei",sans-serif;margin:24px auto;max-width:1700px;padding:0 20px;background:#f4f5f7;color:#18212b}h1{font-size:26px}h2{font-size:19px}section{background:white;padding:18px;margin:28px 0;border:1px solid #ddd}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;margin-top:12px}th,td{border:1px solid #ddd;padding:8px;text-align:center}summary{cursor:pointer}p{line-height:1.7}</style><h1>低高建筑多高度对比</h1><p>'+intro+'</p><p>按z/H=1.05、1.20、1.35、1.50选取最近的可用标签，不按模型分数筛选。点击图片可打开原图。第一行速度，第二行相对扰动，第三行绝对误差；白色为缺测标签。所有高度均高于屋顶。各图最优模型是参考标签已知时的结果，不是自动选模规则。</p>'+''.join(sections)+'</html>',encoding='utf-8')
    archive=ROOT/'deliverables/height_comparisons_24.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as zf:
        for path in sorted(out.iterdir()):
            if path.is_file():zf.write(path,'height_comparisons/'+path.name)
        for name in ['more_comparisons.py','explain_versions.py']:
            zf.write(ROOT/'bench'/name,'scripts/'+name)
    with ZipFile(archive) as zf:assert zf.testzip() is None
    print('DONE '+str(archive),flush=True)


if __name__=='__main__':main()
