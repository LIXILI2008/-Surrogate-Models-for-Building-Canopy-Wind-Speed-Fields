"""Build a new evidence-linked outline revision without altering experiments."""
from pathlib import Path
import json, hashlib, re
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/paper_outline_20261004'
AN=ROOT/'artifacts/prospective_five_region_analysis_20261005_v3'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def link(p,label=None):return f'[{label or p.name}](<{p.as_posix()}>)'
summary=read(AN/'three_seed_summary.json');audit=read(AN/'audit.json')
assert audit['state']=='PASS' and audit['frozen_source_unchanged']
assert sha(ROOT/'analyze_prospective_five_regions.py')==audit['analysis_script_sha256']
methods=['v23_mlp','unet_si','unet','fno','ufno','mlp_data']
regions=['full','roof','windward','side','wake'];cn=['全域','屋顶','迎风','侧缘','尾流']
def val(c,r,m,k='rmse_mps'):return next(x['metrics'][k] for x in summary if x['case']==c and x['region']==r and x['method']==m)
def cell(c,r,m):
    v=val(c,r,m);return f"{v['mean']:.5f} ± {v['sd']:.5f}"
src=BASE/'论文大纲_证据映射版_20261005.md';text=src.read_text(encoding='utf8')
text=text.replace('修订日期：2026-10-05。本版吸收revision_20261005已完成结果；2026-10-04版保留作历史版本。', '修订日期：2026-10-05，v2。本版在同日原修订版基础上，整合新五栋六方法×三种子逐建筑、逐区域冻结评价；此前各版原样保留。')
text=text.replace('## 本版阅读入口与核心调整','## 本版阅读入口与核心调整\n\n本次新增证据为 prospective_five_region_analysis_20261005_v3：18组预测、301个评价高度，450条逐栋区域种子记录（含全域）及90条跨栋宏平均记录。四个目标区域对应360条逐栋记录；所有逐层指标及聚合值复算一致。该分析只整理原冻结评价，不产生新模型或新增独立建筑，也不据此调整模型、掩膜或超参数。\n\n正文优先回答两个相互连接的问题：冻结模型的平均优势在哪些建筑和区域成立；既有少样本适配能否将收益推广到新几何。区域诊断用于描述失败，不用于在五栋上选择最终模型或建立切换规则。')
text=text.replace('### 期刊适配判断','### 期刊适配判断（沿用此前定位，本次未重新核验期刊网页）')
text=text.replace('2026-10-05重查作者指南返回429','此前2026-10-05修订记录称重查作者指南返回429')
text=text.replace('2026-10-05重查Energy and Buildings范围页仍返回403','此前2026-10-05修订记录称重查Energy and Buildings范围页仍返回403')
text=text.replace('随机种子先在对应建筑/实验内汇总；重复种子和相邻切片不能充当独立建筑样本。','冻结六配置结果按每种子先计算逐栋指标，再对五栋等权平均，最后汇总三种子的均值及样本标准差（ddof=1）。对均值，这与先对种子求均值再跨栋平均等价；标准差必须从三个种子的宏平均计算，不能用逐栋标准差的均值代替。重复种子和相邻切片不能充当独立建筑样本。')
anchor='### 2.6.1 数据处理审核与色标敏感性'
insert='''### 2.6.1 冻结区域定义与聚合公式

风向+x，采用协议中的名义W、L、H和既有valid掩膜，不根据新五栋误差或网格测量差值重建掩膜。各区域在128×128裁剪域内定义，并与valid取交集：

|区域|冻结几何定义|解读范围|
|---|---|---|
|屋顶 roof|abs(x)≤L/2，abs(y)≤W/2|屋顶投影上方H<z<1.5H，不等于屋面或首米|
|迎风 windward|−L/2−H≤x<−L/2，abs(y)≤W/2|评价高度内的迎风投影区|
|侧缘 side|abs(x)≤L/2，W/2<abs(y)≤W/2+min(H,W)|建筑两侧区域，不是roof_edge屋顶边缘|
|尾流 wake|L/2<x≤L/2+2H，abs(y)≤max(W,H/4)|几何代理区，不是依据实际回流识别的尾流边界|

区域受共同裁剪域限制，并非全域的穷尽分区；不得将未纳入四区的像素误认为无误差。跨建筑区域物理面积不同，建筑宏平均仍每栋等权。

令r_j=z_j/H，相邻r中点为权重区间边界，首尾延伸至1和1.5，区间宽度归一化得到w_j。每层区域有效像素等权；MSE_b=Σw_j MSE_j，RMSE_b=√MSE_b，MAE_b=Σw_j MAE_j。R²_b=1−MSE_b/[Σw_j E(y_j²)−(Σw_j E(y_j))²]，包含跨高度参考变化；mean_slice_r2=Σw_j R²_j是另一指标。relative_rmse=√Σw_j MSE_j/U(z_j)²，U(z)=3(z/10)^0.4。零方差/缺失值按原代码处理，本次五区域汇总无缺失。

跨栋RMSE为五个RMSE_b的算术平均，不是合并所有像素求RMSE，也不是平均MSE再开方。负R²完整保留。三种子标准差描述训练随机性，不是95%置信区间；本次区域分析没有新增显著性检验或bootstrap。区域结果不增加SSIM：共同13 m/s的全域SSIM仍单独来自revision_20261005，原冻结评价6/13历史SSIM不得混入主表。

证据：analysis_pipeline.py中的region_masks、scalar、height_weights和aggregate_slices，以及新分析的region_coverage.json和audit.json。覆盖表用于追溯有效像素，不据此补填或修改标签。

### 2.6.2 数据处理审核与色标敏感性'''
assert anchor in text;text=text.replace(anchor,insert)
start=text.index('### 3.2 高度、区域和误差图诊断');end=text.index('### 3.3 实验A',start)
section=['### 3.2 新五栋：逐建筑、逐区域与种子波动','',
'先展示全五栋，继而比较区域，再用已有代表层解释；不根据结果删去P05。以下均为严格H<z<1.5H、高度加权、三种子均值±样本标准差，单位m/s。表内均值排序只描述当前五栋，不定义部署模型选择规则。','',
'#### 3.2.1 六方法逐建筑全域RMSE','',
'|方法|P01|P02|P03|P04|P05|跨栋宏平均|','|---|---:|---:|---:|---:|---:|---:|']
for m in methods:section.append('|'+ '|'.join([m,*[cell(c,'full',m) for c in ['P01','P02','P03','P04','P05','MACRO']]])+'|')
section+=['','P01/P02/P03/P04上v23_mlp的三种子全域均值最低；P05上FNO最低。几何范围外不等于必然失败：P04与P05虽都属于轴向范围外，其误差表现不同；不能从五栋拟合高度阈值或网络切换规则。','',
'#### 3.2.2 四区域跨栋宏平均RMSE','', '|方法|屋顶|迎风|侧缘|尾流|','|---|---:|---:|---:|---:|']
for m in methods:section.append('|'+ '|'.join([m,*[cell('MACRO',r,m) for r in regions[1:]]])+'|')
section+=['','v23_mlp在四区域的宏平均RMSE均最低，但逐栋排序有例外：P02侧缘U-Net SI均值更低；P05迎风、侧缘和尾流均为FNO更低，屋顶仍为v23_mlp更低。这些是同一冻结评价的空间差异，不能解释成各模型已识别相应流动物理。','',
'#### 3.2.3 逐栋区域证据的正文与补充安排','',
'正文保留六方法×四区域×五栋的RMSE热力图（另含全域与宏平均面板）及三种子宏平均散点/标准差图；MAE与R²对应图放补充材料。S4提供全部450条逐栋记录、90条宏平均记录及150条逐栋/宏平均×区域×方法三种子汇总中的全部180组（6建筑层级×5区域×6方法），不只保留领先配置。',
'', '|建筑|MLP屋顶|MLP迎风|MLP侧缘|MLP尾流|','|---|---:|---:|---:|---:|']
for c in ['P01','P02','P03','P04','P05']:section.append('|'+ '|'.join([c,*[cell(c,r,'v23_mlp') for r in regions[1:]]])+'|')
section+=['','#### 3.2.4 与已有高度/空间诊断连接','',
'共同高度段1—1.02、1.02—1.1、1.1—1.25、1.25—1.5仍引用已有评价产物；本次新汇总是全高度区域积分，不宣称重新生成了高度分段图。每栋同时展示近1.05H和首种子MLP最大误差层，规则透明。P02/P03/P05所选强误差层的大误差像素主要在尾流代理掩膜，不能推广为所有高度皆如此。',
'', 'P05 z=262 m首种子RMSE约3.653 m/s，参考标准差约1.067 m/s，不能只用低方差解释该层负R²。frame_border仅为评价图外沿6像素，不是CFD域边界。历史U-FNO局部优势不支持H=50 m模型切换。P05参考场质量因素尚未分离，不能把所有差异归因于网络外推。','',
'本节直接证据：'+link(AN/'REPORT.md','区域完整报告')+'、'+link(AN/'three_seed_summary.json','三种子汇总')+'、'+link(AN/'audit.json','复算与源文件审计')+'。','']
section='\n'.join(section).replace('及150条逐栋/宏平均×区域×方法三种子汇总中的全部180组','及全部180组三种子汇总')
text=text[:start]+section+'\n'+text[end:]
text=text.replace('### 4.1 为什么当前MLP具有竞争力','### 4.1 为什么当前MLP具有竞争力，以及宏平均会隐藏什么')
text=text.replace('不同配置训练方式与数据量也是可能因素。','不同配置训练方式与数据量也是可能因素。新五栋四区域宏平均支持其当前配置的竞争力，但P02侧缘和P05多个区域的排序反转表明，不能用全域单一均值概括局部可靠性。区域名称是几何代理，缺少速度矢量、压力和湍流证据时，不写成涡结构重建机理已经得到验证。')
text=text.replace('1. 共同数值评价下，当前MLP配置在两外部队列的RMSE/MAE/SSIM领先；新五栋FNO宏平均R²略高。结论针对当前任务和完整配置。','1. 共同数值评价下，当前MLP配置在两外部队列的RMSE/MAE/共同范围SSIM领先，新五栋四区域宏平均RMSE也最低；FNO新五栋宏平均R²略高，并在P05全域及部分区域更低误差。结论针对当前任务和完整配置，不代表逐栋逐区一致最优。')
text=text.replace('3. 基线：MLP新五栋RMSE 0.59602 m/s，FNO 0.70353 m/s。','3. 基线：MLP新五栋宏平均RMSE 0.59602±0.00659 m/s，FNO 0.70353±0.05578 m/s（±为三种子样本标准差）；P05的FNO全域误差更低。篇幅允许时补充“宏平均领先不等于逐栋逐区领先”。')
text=text.replace('|图3|20栋六方法逐栋/高度误差与新五栋对照|现有all20三图可重排；新五栋用summary.json|','|图3|20栋对照及新五栋六方法逐栋/区域误差、种子波动|新五栋已有rmse_mps_heatmaps.png/svg与macro_seed_spread.png/svg；20栋仍用既有all20图，论文排版待整合|')
text=text.replace('|表3|20栋及新五栋六模型统一指标|各自相同口径汇总，不混屋顶与整场|','|表3|20栋及新五栋六模型统一指标，增加新五栋区域子表|全域SSIM取revision共同13口径；区域误差取最新三种子汇总，均值±标准差，不混屋顶与整场|')
text=text.replace('证据文件索引_20261005.md》提供66个文件','证据文件索引_20261005_v2.md》提供本版全部文件')
text=text.replace('本轮新增证据见第十一节。','本轮新增证据见第十一、十二节。')
text=text.replace('|E12 数值导出|','|E13 新五栋区域冻结分析|analyze_prospective_five_regions.py|prospective_five_region_analysis_20261005_v3/REPORT.md、three_seed_summary.json、audit.json|18组、450条逐栋区域记录及90条宏平均，完整种子波动|\n|E12 数值导出|')
text=text.replace('- S4：全部逐栋、逐高度、逐区域指标及三策略全预算结果。','- S4：全部逐栋、逐高度、逐区域指标及三策略全预算结果。新五栋附building_region_seed_metrics.json、macro_region_seed_metrics.json、three_seed_summary.json、region_coverage.json及RMSE/MAE/R²三图；JSON含未四舍五入值，图表是显示精度。')
text=text.replace('3. 将图1—8按论文逻辑重排，保留物理比例、共同色标和失败案例；图表整理不等于再训练。','3. 新五栋逐栋/区域汇总、种子波动图已经完成；将图1—8按论文逻辑排版，保留物理比例、共同色标和失败案例。仍需新绘图1流程、图2几何覆盖；这些排版待办不属于模型实验。')
text+='''

## 十二、本次区域证据合入、图注与写作核对

本版没有启动训练、适配、重新预测或CFD；新五栋继续只作冻结评价，不用于修改任何模型、掩膜、超参数或选模。A/B适配章节保留此前结果与探索性身份，不以本次区域分析追加适配实验。

**图3新五栋面板建议图注：** 六种冻结模型配置在五个新几何上的高度加权区域误差。行表示配置，列表示全域及四个预定几何区域；P01—P05逐栋展示，MACRO为五栋对应指标等权均值。单元格为三随机种子均值，单位m/s；不同建筑共享色标。评价高度严格H<z<1.5H，各区域与原有效标签掩膜及共同裁剪域相交。预测与标签均为未裁剪标量风速。误差反映与保存CFD解的一致性，不表示实验风场真实性验证。

**种子波动图建议图注：** 彩色点依次表示20261001、20261002、20261003；黑点和误差棒表示宏平均RMSE的三种子均值及样本标准差。每个种子先完成高度加权和五栋等权汇总；误差棒不代表置信区间，五栋之外的推广不确定性未由此估计。

**提交写作前的证据检查：**

|检查项|本版处理|仍需注意|
|---|---|---|
|冻结结果完整性|18组、每组301层；450条逐栋/90条宏平均；复算聚合最大差0|这是算术与产物一致性检查，不是独立实现或物理验证|
|检查点、预测及标签|最新分析记录SHA256及映射审核通过|本次大纲更新复用该审计，不声称重新打开Fluent或复核全部CFD源文件|
|源文件保护|新分析审计确认冻结源文件前后哈希一致；本次仅更新写作产物|旧大纲和旧索引保留；README指向本版|
|区域完整性|四目标区域及全域均报告，包括全部失败值|不能根据目标结果改区界、删建筑或挑种子|
|统计单位|5栋，3种子为重复实现；高度按间距加权|不称15栋或301个独立验证样本|
|SSIM|主表沿用revision共同13全域结果；新区域分析未计算SSIM|不可拿原冻结6/13值替换主表|
|物理边界|convergence_certified=false；无实验真实性认证|正文、摘要、图注与结论保持一致|
|证据索引|本版重新检查所列文件存在性、SHA256及与旧索引差异|哈希变化必须留痕，不自动推断为数据损坏|

本次不重新核验文献、期刊范围或投稿格式；原文献矩阵和期刊定位仅沿用此前记录。正式撰稿时引用论文原文并核对期刊现行要求，不将本次文字完善称为已完成文献系统审查。
'''
dest=BASE/'论文大纲_证据映射版_20261005_v2.md';assert not dest.exists();dest.write_text(text,encoding='utf8')
old=read(BASE/'evidence_index_20261005.json');paths=[Path(r['path']) for r in old]
paths += [ROOT/'analyze_prospective_five_regions.py',ROOT/'update_paper_outline_regions.py',*sorted(AN.iterdir()),dest]
index=[];prev={r['path']:r.get('sha256') for r in old}
for p in dict.fromkeys(paths):
    exists=p.is_file();h=sha(p) if exists else None
    index.append(dict(path=p.as_posix(),exists=exists,sha256=h,previous_sha256=prev.get(p.as_posix()),changed_from_previous=(h!=prev[p.as_posix()]) if p.as_posix() in prev else None))
assert all(r['exists'] for r in index)
(BASE/'evidence_index_20261005_v2.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# 2026-10-05 v2证据索引','','本次对所列文件重新检查存在性并计算SHA256。未重跑模型或CFD；仅核对写作引用。完整哈希见同名JSON。', '',f'共{len(index)}个文件，全部存在；与上一索引相比改变{sum(r["changed_from_previous"] is True for r in index)}个。新增记录不算变更。','','|文件|存在|相对旧索引|','|---|---|---|']
for r in index:lines.append('|'+link(Path(r['path']),str(Path(r['path']).relative_to(ROOT.parent)))+'|是|'+('新增' if r['changed_from_previous'] is None else '变化' if r['changed_from_previous'] else '一致')+'|')
(BASE/'证据文件索引_20261005_v2.md').write_text('\n'.join(lines),encoding='utf8')
(BASE/'README.md').write_text('# 论文大纲入口\n\n当前写作依据：'+link(dest,'2026-10-05 v2：新五栋逐建筑/逐区域证据完善版')+'。\n\n- '+link(BASE/'证据文件索引_20261005_v2.md','当前证据索引')+'\n- '+link(BASE/'evidence_index_20261005_v2.json','机器可读索引')+'\n- '+link(AN/'REPORT.md','新五栋冻结区域分析报告')+'\n\n原2026-10-04版及2026-10-05首版均保留。本版补齐六方法三种子的逐栋全域对照、四区域宏平均、统计公式、图注与证据边界；未启动训练、调参或CFD。\n',encoding='utf8')
# Validate all absolute local file links in the new outline.
links=re.findall(r'\]\(<(C:/[^>]+)>\)',text)
assert all(Path(x).exists() for x in links)
print(json.dumps(dict(outline=str(dest),indexed_files=len(index),changed_existing=[r['path'] for r in index if r['changed_from_previous']],local_links_verified=len(links)),ensure_ascii=False))
