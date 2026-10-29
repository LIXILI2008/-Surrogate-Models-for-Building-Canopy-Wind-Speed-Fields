from pathlib import Path
import sys,json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deliverables/chapter3_plot_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import font_manager
from windscale_geometry import annotate_points

OUT=ROOT/'project_v2_5/artifacts/windscale_geometry_20261016'
s=json.loads((OUT/'summary.json').read_text(encoding='utf8'))
rows=s['buildings'];twenty=[r for r in rows if r['cohort']=='retrospective20'];five=[r for r in rows if r['cohort']=='corrected5']
font=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc');font_manager.fontManager.addfont(font.get_file())
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(7.5,4.6),layout='constrained')
x=np.arange(20)
ax.bar(x,[r['nrmse_pct'] for r in twenty],color=['#2d6d9b' if r['H']<=50 else '#d58b30' for r in twenty])
ax.set_xticks(x,[r['id'] for r in twenty],rotation=90);ax.set_ylabel('按入口风速归一化的误差（%）');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
for group,color in [('D：低建筑组','#2d6d9b'),('G：高建筑组','#d58b30')]:ax.plot([],[],color=color,lw=7,label=group)
ax.legend(loc='upper left',frameon=False,fontsize=9)
for ext in ['png','svg']:fig.savefig(OUT/'figures'/f'05_normalized_by_building.{ext}',dpi=300)
plt.close()
fig,ax=plt.subplots(figsize=(7.5,5.1),layout='constrained')
sc=ax.scatter([r['nearest_distance'] for r in twenty],[r['nrmse_pct'] for r in twenty],c=[r['H'] for r in twenty],cmap='viridis',s=55,edgecolors='#333',lw=.5)
ax.set(xlabel='与最近训练建筑的尺寸距离',ylabel='按入口风速归一化的误差（%）');ax.grid(alpha=.2);ax.margins(.16)
ax.set_ylim(0,100)
ax.set_yticks(np.arange(0,101,10))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100,decimals=0))
fig.colorbar(sc,ax=ax,label='建筑高度 H（m）',pad=.02)
annotate_points(ax,twenty,'nearest_distance','nrmse_pct')
for ext in ['png','svg']:fig.savefig(OUT/'figures'/f'06_distance_normalized.{ext}',dpi=300)
plt.close()

def image(name,title):return f'\n![{title}]({(OUT/"figures"/name).as_posix()})\n\n**图 {title}**\n'
METHODS={'v23_mlp':'MLP','unet_si':'U-Net SI','unet':'U-Net','fno':'FNO','ufno':'U-FNO','mlp_data':'MLP-data'}
lines=['# 风速尺度与训练尺寸覆盖分析','',
'## 1 分析范围与主要发现','',
'本次分析使用V2.5登记的2026年10月9日W32修订批次结果：原28栋训练，20栋作回顾性评价，修订五栋单独评价。六种配置各使用三个训练种子。计算对象为原模型，未更新权重或加入新的训练建筑。以下几何分析以MLP为主。','',
'主要发现是：高低建筑的误差差距在按入口风速归一化后明显缩小；建筑尺寸偏离训练样本的程度与误差有一定关联，但关联随距离定义和建筑组别变化，不能据此划定可靠与不可靠的尺寸阈值。','',
'## 2 计算方法','',
'### 2.1 按每层入口风速归一化','',
'沿用数据集与评价代码中的入口风速公式Uin(z)=3(z/10)^0.4 m/s。每张切片先将风速误差除以其高度z对应的Uin，再计算平方误差；同一建筑按原有高度间距权重汇总，最后开平方。写为：','',
'归一化误差（%）=100×√[Σⱼwⱼ×MSEⱼ/Uin(zⱼ)²]。','',
'MSEⱼ为该层共同有效位置上的平均平方误差，wⱼ为该层代表的高度区间权重。先计算每次训练的建筑级指标，再平均三个种子的结果；组内建筑等权。评价范围保持H<z<1.5H。此处的百分比以入口背景风速为尺度，不是逐点百分比误差，也没有除以可能接近零的局部参考风速。0—6和0—13 m/s的图像色标不参与本次归一化。','',
'逐切片文件中原本已有relative_mse字段，本次按照公式重新核算并汇总。36组评价文件共21960条含种子、模型重复的切片记录，重算归一化平方误差与保存值的最大差为0，重算原RMSE与共同评价主表的最大差为0。','',
'### 2.2 与原28栋训练尺寸的距离','',
'原28栋的实际登记范围为W=8—30 m、L=4—72 m、H=4—100 m，其中低建筑23栋、高建筑5栋。W始终为垂直来流方向的宽度，L为沿来流方向的长度，不交换长宽。采用两种互补描述。','',
'第一种是超出范围的距离：分别计算W、L、H超出上述区间的部分，再除以各自跨度22、68、96 m，最后计算三项平方和的平方根。完全位于三个区间内时距离为0；这仅表示各尺寸未越界，不表示训练中存在相近的尺寸组合。','',
'第二种是最近训练建筑距离：对每栋训练建筑计算√[((W−Wᵢ)/22)²+((L−Lᵢ)/68)²+((H−Hᵢ)/96)²]，取最小值。它同时保留三项尺寸的组合信息，与V2.5原有nearest_normalized_distance的定义一致。数值无量纲，越大表示按此尺度衡量时距离训练样本越远。','',
'另将W、L、H取自然对数、按原28栋对数范围缩放后重算最近距离，以检查结论是否依赖距离的定义。该定义更侧重尺寸倍数差异；它是敏感性分析，没有根据误差挑选更有利的定义。','',
'### 2.3 不确定性与解释范围','',
'相关性采用Spearman等级相关系数ρ，比较建筑间的排序：正值表示总体上一个量越大，另一个量也越大；负值表示总体上反向变化。使用20栋的逐栋值计算，不把切片或三个种子当成独立建筑。相关系数区间来自2000次建筑重抽样，高低组均值差来自各组内5000次建筑重抽样；这些区间描述当前案例的重抽样稳定性，不代表随机抽取的城市建筑总体。另报告每次去掉一栋后的相关性范围，并检查高度排序调整后的关联。','',
'本次分析在已查看20栋结果之后开展，属于回顾性补充分析。相关性不是因果效应，多项探索性比较不能据此确立普遍规律或工程阈值。','',
'## 3 风速归一化结果','',
'### 3.1 高低组的差距缩小','',
'**表1 MLP在20栋中的高低组比较**','',
'|指标|低建筑10栋|高建筑10栋|高组/低组|高组减低组的95%重抽样区间|',
'|---|---:|---:|---:|---|',
f"|RMSE（m/s）|{s['group_comparison']['rmse_mps']['low_mean']:.4f}|{s['group_comparison']['rmse_mps']['high_mean']:.4f}|{s['group_comparison']['rmse_mps']['high_low_ratio']:.2f}|0.1725—0.4804 m/s|",
f"|归一化误差（%）|{s['group_comparison']['nrmse_pct']['low_mean']:.2f}|{s['group_comparison']['nrmse_pct']['high_mean']:.2f}|{s['group_comparison']['nrmse_pct']['high_low_ratio']:.2f}|−0.20—2.81个百分点|",'',
'高低组的绝对误差相差约2.19倍，归一化后相差约1.23倍。高建筑的归一化均值仍较高，但均值差的重抽样区间包含0。因此，本次数据不支持仅凭绝对RMSE的差距，就认定高建筑普遍更难预测。也不能把比值缩小的幅度当作背景风速的因果贡献比例。','',
'建筑高度与绝对RMSE的等级相关系数为0.871，归一化后降至0.558。高度与误差的连续关联仍然存在，但“高度越高的排序趋势”和“以50 m划分两组的均值差”是不同问题；两项结果并不矛盾。',
image('05_normalized_by_building.png','1 20栋建筑的归一化误差'),
'### 3.2 部分建筑的误差排名发生变化','',
'D10的绝对RMSE为0.5137 m/s，低于G06的0.8237 m/s；但归一化误差分别为8.90%和7.71%，排序发生反转。G07与G03的归一化误差仍较高，分别为10.15%和9.05%。这说明绝对误差较小并不一定意味着相对于背景风速的偏差也小。','',
'修订五栋中，P05的绝对误差最高，为0.7159 m/s；归一化后误差最高的则是P04，为7.95%，P05为6.44%。这里没有修改预测或参考数据，只改变了观察误差的尺度。','',
'### 3.3 六种配置的总体排序','',
'**表2 六种配置的建筑平均归一化误差**','',
'|配置|20栋 RMSE（m/s）|20栋 归一化误差（%）|修订五栋 RMSE（m/s）|修订五栋 归一化误差（%）|','|---|---:|---:|---:|---:|']
for key,name in METHODS.items():
 a=next(r for r in s['group_summaries'] if r['method']==key and r['cohort']=='retrospective20' and r['group']=='all');b=next(r for r in s['group_summaries'] if r['method']==key and r['cohort']=='corrected5' and r['group']=='all')
 lines.append(f"|{name}|{a['rmse_mps']:.4f}|{a['nrmse_pct']:.2f}|{b['rmse_mps']:.4f}|{b['nrmse_pct']:.2f}|")
lines+=['','MLP在两组建筑的归一化误差仍最低，分别为6.32%和5.91%。因此，本次归一化没有改变MLP作为当前总体基线的结论。修订五栋中FNO与U-Net的相对次序发生变化，表明次优配置的排序也受误差尺度影响。','',
'按原有高度段计算，20栋MLP在z/H为1.00—1.02、1.02—1.10、1.10—1.25、1.25—1.50的归一化误差分别为12.25%、10.37%、6.18%、2.39%。按区域计算，建筑前方为3.41%，两侧为5.07%，轮廓正上方为6.73%，下游为9.33%。因此，近屋顶和下游误差较大的现象在归一化后仍然保留。','',
'## 4 训练尺寸覆盖与误差','',
'### 4.1 尺寸越界不等于预测失败','',
'20栋中13栋至少有一项尺寸超出原28栋范围，7栋位于三个尺寸区间内；修订五栋中3栋超出范围。这里不将各尺寸区间内的案例称为已被训练覆盖，因为训练尺寸组合是稀疏的。','',
'D08与最近训练建筑的尺寸距离为2.032，是20栋中最大的，但归一化误差为5.10%；D10的距离仅为0.270，且各尺寸都未超出范围，归一化误差却为8.90%。因此，尺寸范围和最近距离可用来标记训练覆盖不足，不能单独判定预测是否可靠。','',
'### 4.2 最近训练建筑距离与误差有一定关联','',
'**表3 20栋MLP归一化误差与尺寸特征的关联**','',
'|尺寸描述|等级相关系数ρ|95%重抽样区间|每次去掉一栋后的ρ范围|','|---|---:|---|---|']
labels={'H':'建筑高度H','WH':'宽高比W/H','LH':'长高比L/H','box_distance':'超出范围的距离','nearest_distance':'最近训练建筑距离','nearest_log_distance':'对数尺寸最近距离'}
for c in s['correlations']:
 if c['metric']=='nrmse_pct':
  ci=c['bootstrap_ci95'];loo=c['leave_one_out_range'];lines.append(f"|{labels[c['feature']]}|{c['rho']:.3f}|{ci[0]:.3f}至{ci[1]:.3f}|{loo[0]:.3f}至{loo[1]:.3f}|")
lines+=['','最近训练建筑距离与归一化误差的ρ为0.502，每次去掉一栋后为0.419—0.616，表示本批建筑中存在距离较大、误差也偏大的总体倾向。仅计算超出范围的距离时，ρ为0.395，区间包含0，证据较弱。','',
'但将尺寸改为对数尺度后，最近距离与归一化误差的ρ降至0.248，区间为−0.215至0.626。因此，这一关联对距离定义敏感，不能据此写成“几何外推导致误差增加”的确定结论。','',
'将高低组分别计算时，最近距离与归一化误差的ρ在低组为0.430，在高组为0.697。两组各只有10栋，区间较宽，且本次为事后检查，应作为补充证据。宽高比与归一化误差的原始ρ为−0.460，在调整建筑高度排序后为−0.031；这表明宽高比的表面关联与高度变化明显重叠，不能据此建立独立的宽高比规律。',
image('06_distance_normalized.png','2 最近训练建筑距离与归一化误差（纵轴0%—100%）'),
'## 5 逐栋结果','',
'**表4 MLP逐栋误差与训练尺寸覆盖**','',
'|编号|H（m）|RMSE（m/s）|归一化误差（%）|越界尺寸|最近距离|','|---|---:|---:|---:|---|---:|']
for r in twenty+five:lines.append(f"|{r['id']}|{r['H']:g}|{r['rmse_mps']:.4f}|{r['nrmse_pct']:.2f}|{r['outside_axes'].replace('none','无')}|{r['nearest_distance']:.3f}|")
lines+=['','注：D为低建筑组，G为高建筑组，P为修订五栋。高低组按H≤50 m和H>50 m划分；原始数据中的L、U编号分别对应论文D、G编号。上述误差均为原模型结果。','',
'## 6 对论文表述的影响','',
'### 6.1 可替换第三章原逐栋分析的文字','',
'为区分风速尺度与建筑尺寸的影响，进一步按每张切片高度对应的入口风速归一化误差。MLP在低、高建筑组的绝对RMSE分别为0.2712和0.5944 m/s，高组约为低组的2.19倍；归一化后分别为5.68%和6.97%，比值降至1.23。高低组归一化均值差的95%重抽样区间为−0.20至2.81个百分点，包含0。这表明仅用绝对RMSE比较高低建筑，会明显受到风速尺度的影响。近屋顶和下游误差较大的现象在归一化后仍然存在，MLP在两组评价中的总体优势也保持不变。','',
'原28栋的训练尺寸为W=8—30 m、L=4—72 m、H=4—100 m。20栋中有13栋至少一项尺寸超出该范围。按训练尺寸跨度缩放后，建筑与最近训练样本的距离与归一化误差呈正向关联，等级相关系数为0.502，但对数尺寸距离下仅为0.248。D08距离训练样本最远，归一化误差为5.10%；D10各尺寸均未超出范围，误差却达到8.90%。因此，训练尺寸覆盖不足可作为需要额外核查的线索，但不能单凭尺寸越界或几何距离判断模型的适用性。','',
'### 6.2 讨论中应保留的限制','',
'归一化是改变误差的比较尺度，不是模型精度提升，也不改变原预测、R²或SSIM。高组仅有5栋源训练建筑，训练密度、几何比例、入口风速及标签来源存在交织，本次相关分析不能拆分各因素的因果贡献。历史低组训练标签含图像反解，而本次20栋与修订五栋使用数值标签；训练和评价物理视野不同，CFD参考本身也存在不确定性。','',
'修订五栋仅逐栋展示，不并入20栋拟合关系或挑选距离定义；它们未参与本轮学习，但曾经查看结果及修订案例，不能把本次事后分析再称为新的盲测。当前没有给出可直接用于工程决策的误差上限或尺寸阈值。','',
'## 7 可复现文件','',
'脚本：project_v2_5/scripts/windscale_geometry.py。使用项目Python直接运行该脚本，可读取config.json登记的历史结果，在project_v2_5/artifacts/windscale_geometry_20261016下生成独立分析。','',
'building_metrics_and_geometry.csv：25栋逐栋误差、尺寸、越界程度及最近训练建筑；six_model_group_summary.csv：六配置高低组及整体统计；seed_region_height_metrics.csv：逐种子、区域、高度段指标；training28_dimensions.csv：训练尺寸；summary.json：重抽样区间与相关分析；provenance.json：全部输入来源及SHA256。图像同时保存PNG和SVG。']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
(OUT/'论文3.3.3补充段落.md').write_text('\n\n'.join([lines[lines.index('### 6.1 可替换第三章原逐栋分析的文字')+2],lines[lines.index('### 6.1 可替换第三章原逐栋分析的文字')+4]]),encoding='utf-8')
print(OUT/'REPORT.md')
