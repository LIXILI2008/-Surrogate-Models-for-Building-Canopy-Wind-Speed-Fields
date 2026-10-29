"""Reproduce two matched-grid illustrations without changing experiment results."""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .common import ROOT, WORKSPACE, read, write, digest, inlet
from .data import Dataset
from .models import make, CoordinateModel
from .metrics import score

OUT=ROOT/'deliverables/version_explanation'
EXP=ROOT/'artifacts/comparison'
V23=WORKSPACE/'project_v2_3/artifacts/height_transfer_numeric_verified'
METHODS=['v23_reference','unet','fno','ufno','pinn']
LABELS=['V2.3','V2.4 U-Net','V2.4 FNO','V2.4 U-FNO','V2.4 PINN']


def examples(plans=None):
    sys.path.insert(0,str(WORKSPACE/'project_v2_3'))
    from fieldgen.train import load
    from fieldgen.predict import infer
    ds=Dataset(ROOT/'artifacts/dataset_128')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    if plans is None:
        plans=[('bldg_W16_L20_H17',19.,'low_H17_z19'),('bldg_W24_L8_H100',110.,'high_H100_z110')]
    for name,z,slug in plans:
        print(f'COMPARISON {name} z={z:g}',flush=True)
        idx=next(i for i in ds.ids[name] if abs(ds.rows[i]['z']-z)<1e-6)
        row=ds.rows[idx];b=ds.batch([idx],device);q=b['q'][0].cpu().numpy()
        valid=ds.valid[idx];dtrue=ds.target[idx,0].copy();dims=row['dims'];u0=inlet(z)
        truth=(dtrue+1)*u0;predictions=[];scores={};building={};proof=[]
        for method in METHODS:
            if method=='v23_reference':
                cp=V23/'low_regression'/name/'fit/best.pt' if dims[2]<=50 else V23/'folds/joint_balanced'/name/'best.pt'
                model,ck=load(cp,device)
                with torch.no_grad():d=infer(model,q[...,:2]*dims[2],dims,z,device)/u0-1
            else:
                cp=EXP/method/name/'best.pt';ck=torch.load(cp,map_location='cpu',weights_only=True)
                model=make(ck['architecture']).to(device);model.load_state_dict(ck['state']);model.eval()
                with torch.no_grad():
                    if isinstance(model,CoordinateModel):
                        f=b['f'][0].reshape(-1,17)
                        output=torch.cat([model.from_features(v) for v in f.split(8192)]).reshape(128,128,7)
                    else:output=model(b['f'].permute(0,3,1,2))[0].permute(1,2,0)
                    d=output[...,0].cpu().numpy()
            assert name not in ck['trained_buildings']
            measured=score(dtrue,d,valid,q,dims,z)
            saved=next(s for s in read(EXP/method/name/'per_slice.json') if abs(s['z']-z)<1e-6)
            assert abs(measured['speed']['rmse']-saved['speed']['rmse'])<1e-5
            scores[method]=saved;building[method]=read(EXP/method/name/'metrics.json')
            proof.append(dict(method=method,checkpoint=str(cp),sha256=digest(cp),held_out_excluded=True))
            predictions.append((d+1)*u0)
        fig,axes=plt.subplots(3,6,figsize=(17,8.5),layout='constrained')
        all_speed=[truth,*predictions];all_d=[dtrue,*[(p/u0-1) for p in predictions]]
        vmax=6 if dims[2]<=50 else 13;emax=.5 if dims[2]<=50 else 1.5
        extent=[row['bounds'][0],row['bounds'][2],row['bounds'][3],row['bounds'][1]]
        for j,(title,v,d) in enumerate(zip(['Reference label',*LABELS],all_speed,all_d)):
            speed_im=axes[0,j].imshow(np.where(valid,v,np.nan),cmap='turbo',vmin=0,vmax=vmax,
                                    interpolation='nearest',extent=extent)
            d_im=axes[1,j].imshow(np.where(valid,d,np.nan),cmap='coolwarm',vmin=-.5,vmax=.5,
                                 interpolation='nearest',extent=extent)
            error=abs(v-truth)
            err_im=axes[2,j].imshow(np.where(valid,error,np.nan),cmap='magma',vmin=0,vmax=emax,
                                   interpolation='nearest',extent=extent)
            axes[0,j].set_title(title,fontsize=12)
            if j:
                s=scores[METHODS[j-1]]['speed']
                axes[2,j].set_xlabel(f"RMSE {s['rmse']:.3f} m/s\nSlice R2 {s['r2']:.3f}",fontsize=10)
            else:axes[2,j].set_xlabel('Reference minus itself',fontsize=10)
            for ax in axes[:,j]:
                ax.set_xticks([]);ax.set_yticks([])
                ax.add_patch(Rectangle((-dims[1]/2,-dims[0]/2),dims[1],dims[0],
                                       fill=False,edgecolor='black',linestyle='--',linewidth=.65))
        for ax,label in zip(axes[:,0],['Speed (m/s)','D = U/Uin - 1','Absolute error (m/s)']):ax.set_ylabel(label,fontsize=12)
        for ax,im,label in zip(axes,[speed_im,d_im,err_im],['m/s','dimensionless','m/s']):
            fig.colorbar(im,ax=list(ax),fraction=.014,pad=.01,label=label,extend='max' if label=='m/s' else 'both')
        kind='calibrated image-derived label' if dims[2]<=50 else 'accepted Fluent numeric label'
        fig.suptitle(f'W={dims[0]:g}, L={dims[1]:g}, H={dims[2]:g} m; z={z:g} m; Uin={u0:.3f} m/s\n'
                     f'{kind}; same 128x128 grid; whole-building holdout; wind along +x',fontsize=13)
        fig.supxlabel('Dashed rectangle: building projection, valid fluid above roof. White: unavailable labels. Fixed color limits; values outside limits are saturated.',fontsize=10)
        fig.savefig(OUT/(slug+'.png'),dpi=150);plt.close(fig)
        np.savez_compressed(OUT/(slug+'.npz'),truth_mps=truth,predictions_mps=np.stack(predictions),
                            methods=np.array(METHODS),valid=valid,q=q,dims=dims,z=z)
        rows.append(dict(building=name,dims=dims,z=z,inlet_mps=u0,figure=slug+'.png',single_slice=scores,
                         all_heights=building,provenance=proof))
    write(OUT/'examples.json',rows)
    return rows


def fmt(v):return '未定义' if v is None else f'{v:.4f}'


def write_report(rows):
    summary=read(EXP/'summary.json')['groups']
    v23summary=read(WORKSPACE/'project_v2_3/deliverables/summary.json')
    meta=read(ROOT/'artifacts/dataset_128/manifest.json')
    all_methods=['v23_reference','unet_si','unet','fno','ufno','mlp_data','pinn']
    parts=[
        '# V2.3与V2.4详细说明及低/高建筑实例',
        '本说明根据当前代码、已保存留出评估和重新加载权重后的同网格推理编制。未重新训练、未改写原实验指标。',
        'V2.3是已选定的几何条件坐标MLP生产方案。V2.4是相对扰动标签与多架构研究对比，包含6种配置、7个整建筑留出，共42次训练；没有自动替换V2.3生产模型。',
        '## 共同任务与数据',
        'W是横风宽度，L是顺风长度，H是建筑高度，z是离地高度。目标为H≤z≤1.5H的速度大小图；不是仅凭彩色速度图恢复完整矢量场。入口Uin(z)=3(z/10)^0.4 m/s。实际数据范围：W=8～30m，L=4～72m，H=4～100m；范围内任意组合不等于有充分训练支持。',
        '数据共28种建筑：23个低建筑、771张既有色表校准图像标签；5个高建筑、220张已有Fluent数值切片。高建筑每1m导出，覆盖H～1.5H；低建筑高度采样不均匀，部分有0.1m密集切片，也有1m间隔或顶部缺片。低建筑并未重新导出数值真值。',
        '高建筑4个采用1200次迭代结果，W24×L72×H80采用798次已达到本轮原残差标准的结果。1200次是接受的迭代基准，不能写成全部严格收敛。',
        '建筑投影是输入几何条件；z<H的固体屏蔽、z=H的屋顶零速边界、z>H的投影内流体必须分开。z=H预设零速蓝色屋顶排除流体评分。白色缺测像素是标签不可用区域，不能据此认定预测中的实体范围。',
        '## V2.3流程与模型',
        '原始低建筑校准标签/高建筑数值标签 → SI速度和几何审计 → 高建筑512网格及每切片4096点缓存 → 建筑/高度/空间区域分层抽样 → 条件坐标MLP全局训练 → 局部头微调 → 整建筑留出评估 → joint_balanced全部28栋生产重训 → 任意实数z的512网格查询。',
        '输入为25个位置/距离/投影/高度特征和5个尺寸条件。位置特征含三个频率的正弦余弦编码；条件为log(W/16)、log(L/24)、log(H/20)、log(W/H)、log(L/H)。主干宽96，4个线性隐藏层，SiLU；条件支路5→64→192，产生缩放和平移；局部支路30→64→64→1。总参数49,602。没有建筑ID查表，也没有U-Net编码器解码器。',
        '输出可写为Uhat=Uin(z)+DeltaU。实现采用以U(H)为尺度的softplus正值参数化，初始输出恰为入口速度；局部残差头由边界/近屋顶门控调节。',
        '训练损失等价于 sum[w*((Uhat-U)/U(H))²]/sum[w]，其中w=1+4*边界权重+6*屋顶邻域权重。虽然速度点缓存统一除以13一次，但损失已进一步按U(H)归一化；不能说V2.3完全没有无量纲处理。',
        '选中joint_balanced：低/高域各50%监督点，再在域内按建筑、5个相对高度区间及投影/边界/尾流/其余区域抽样。4000全局步+300局部步，每步4096点。AdamW，全局学习率0.0007，局部0.00005，weight decay=0.0001，余弦退火，梯度裁剪1。种子20260918。局部微调是训练建筑共用的局部支路训练，预测新建筑时不需要其真值。',
        '显式迁移方案已比较：低建筑预训练2500步，随后高建筑混合回放适配1500步+局部300步。两种迁移候选未胜过均衡联合训练，因此生产模型通过联合共享主干使用低建筑信息，而非选择单独预训练再迁移。',
        '## V2.4标签与网络',
        '主标签D=U/Uin(z)-1；反变换U=Uin(z)*(1+D)，所有新模型初始D=0。该表达可逆，没有把真值改成更容易预测的数字。最大标签反变换误差小于1e-6 m/s，不能消除原图颜色校准、配准或CFD残差误差。',
        'D≤−0.1为减速至少10%，D≥0.1为加速至少10%；20%作为固定敏感性阈值。尾流ROI为L/2≤x≤L/2+2H且|y|≤max(W,0.25H)，与有效视野相交；ROI与预测无关。保留IoU@4，同时报告区域占比、空类别、误报率、扰动R²及尾流RMSE。',
        '全部新模型使用17个解析位置/几何/入口通道，输出7通道：D、u/U(H)、v/U(H)、w/U(H)、p/(rho*U(H)²)、k/U(H)²、epsilon*H/U(H)³。高建筑提供真实辅助数值监督；低建筑只有速度大小。速度标量与插值矢量的模不是同一插值量，因此保留独立标量头。',
        '| 模型 | 实现 | 参数量 |\n|---|---|---:|\n| U-Net / unet_si | 17输入，16→32→64→128通道，三级池化、跳跃连接、GroupNorm、GELU | 491432 |\n| FNO | 宽20，4层二维谱卷积，每方向8模态，8像素补零，1×1局部分支 | 412396 |\n| U-FNO | 上述FNO最后两层并联基础宽8的U-Net分支 | 660820 |\n| mlp_data / PINN | 17输入、宽128、5隐藏层、Tanh、7输出；附局部标量支路 | 70472 |',
        'U-Net用相邻像素与多尺度跳跃连接形成图像；FNO用频域卷积实现全图信息传递；U-FNO增加局部多尺度支路。它们均为本任务的二维切片适配：高度作为条件输入，不是输入整个高度序列的三维网络。PINN为连续三维坐标网络，用z方向导数施加方程约束。',
        '统一128×128标签网格，双线性标量插值只保留有效支持率>0.999的位置。每步抽一个低建筑切片、一个高建筑切片，共8192候选监督位置；实际损失只计有效像素，域内建筑均衡，z/H分3段抽样。3000全局步+300局部步，AdamW学习率0.0005/0.00005，余弦退火，weight decay=0.0001，梯度裁剪1，种子20260911。不同模型参数量与计算时间不同。',
        '新损失包含几何加权D均方误差、三类区域的有限幅度频率平衡、系数0.002的10%区域软分类损失，以及高建筑共享辅助监督（u/v/w/p为0.1，k/epsilon变换误差为0.01）与屋顶矢量零速约束。unet_si用同结构/辅助数据，只改用速度/13误差并关闭新分区/软分类项；它是标签与损失组合消融，不是单独归一化消融。',
        'PINN在相同MLP和监督数据上增加三维稳态不可压缩标准k-epsilon RANS：连续性、三个动量方程、k和epsilon输运方程。仅在训练高建筑内部离墙>2m处每步96个点，200步后逐渐启用，到1000步达到系数0.001。实际case核验的常数为rho=1.225、mu=1.7894e-5、Cmu=0.09、C1=1.44、C2=1.92、sigma_k=1、sigma_epsilon=1.3。',
        'PINN没有复现Fluent壁面函数，也没有将裁剪框当作真实入口边界。PDE直接约束矢量、压力及湍流辅助输出，对速度大小头主要通过共享网络间接影响。低建筑矢量缺乏监督，因此不据此宣称其矢量场已验证。',
        '两版均可直接查询指定实数z，不必先生成整个高度范围再挑图。V2.4没有显式的相邻切片序列损失；连续高度条件本身不保证更准确。',
        '## 统一网格的结果（避免混用原512网格分数）',
        '以下V2.3参考分数来自其原留出权重在同128网格的重新评估；V2.4是新训练模型。评估口径相同，训练结构/预算/辅助数据仍不同。低组只含2栋，高组只含4栋范围内高建筑；L72留一时超出其余训练长度范围而单列。',
    ]
    for group,title in [('low_regression','低建筑2栋'),('high_in_range','高建筑范围内4栋')]:
        lines=[f'### {title}','','| 模型 | RMSE m/s | R² | 逐图R² | SSIM | IoU@4 | 减速IoU10% | 尾流RMSE |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in all_methods:
            m=summary[method][group]
            lines.append('| '+method+' | '+' | '.join(fmt(m[k]) for k in ['rmse_mps','r2','mean_slice_r2','ssim','iou_at_4','deficit_iou10','wake_rmse_mps'])+' |')
        parts.append('\n'.join(lines))
    parts.append('R²先在单建筑合并全部高度的有效像素计算，再跨建筑平均；逐图R²是另一种口径。合并高度引入入口速度变化，可能掩盖单张图误差。SSIM固定速度范围：低6、高13 m/s；128与512网格的像素窗口物理尺度不同，不能混算。')
    parts.append('原V2.3原网格全图：低RMSE0.0987/R²0.9672/SSIM0.9909；高RMSE0.2579/R²0.9531/SSIM0.9919。它们与上面的128网格分数差异来自重采样及指标计算网格，不是模型重新训练。')
    coverage=[]
    for method in all_methods:
        ms=[read(EXP/method/n/'metrics.json') for n in ['bldg_W16_L20_H17','bldg_W8_L24_H12']]
        coverage.append(dict(method=method,iou_at_4_defined_buildings=sum(m['iou_at_4']['iou'] is not None for m in ms),
                             deficit10_defined_buildings=sum(m['deficit_0.1']['iou'] is not None for m in ms)))
    write(OUT/'iou_coverage.json',coverage)
    parts.append('IoU宏平均只计算有定义的建筑：V2.3低建筑IoU@4=0.4501包含2栋；U-Net的0.7926只有1栋有定义（另一栋真实和预测均无≥4区）。因此不能将0.4501→0.7926单独称作泛化提升。新增相对区域指标仍需要同时查看类别是否存在。空并集不能强制当作完美分数。')
    for example in rows:
        name=example['building'];dims=example['dims'];z=example['z'];u0=example['inlet_mps']
        parts.extend([f'## 实例：W{dims[0]:g}×L{dims[1]:g}×H{dims[2]:g}，z={z:g}m',
            f'入口速度为{u0:.4f}m/s。减速10%对应U≤{.9*u0:.4f}m/s，加速10%对应U≥{1.1*u0:.4f}m/s。'
            '所有示例权重均排除了该建筑的全部高度切片；因此是整建筑留出预测，不是生产模型对训练样本回放。',
            '下面表格仅对应这一张切片，不能与上面跨建筑/高度汇总分数混用。'])
        lines=['| 模型 | 单图RMSE m/s | 单图R² | SSIM | 减速IoU10% | 尾流RMSE |','|---|---:|---:|---:|---:|---:|']
        for method in METHODS:
            m=example['single_slice'][method]
            vals=[m['speed']['rmse'],m['speed']['r2'],m['ssim_speed'],m['deficit_0.1']['iou'],m['geometric_wake']['rmse']]
            lines.append('| '+method+' | '+' | '.join(map(fmt,vals))+' |')
        parts += ['\n'.join(lines),f"![同网格速度、扰动和误差对比]({example['figure']})",
            '图中从左至右：参考标签、V2.3、U-Net、FNO、U-FNO、PINN；上排速度，中排相对扰动，下排绝对误差。黑虚线只是投影轮廓，z>H时其内部仍属流体。白色表示标签不可用。颜色上下限固定，超范围会饱和；精确数值见配套NPZ和JSON。']
        if dims[2]<50:
            parts.append('本例V2.3整图RMSE最低，但U-Net的减速区IoU和尾流RMSE略优。二者都恢复了明显的屋顶上方减速区；FNO/U-FNO及PINN仍有局部误差。该建筑目标范围17～25.5m，已有27张参考标签最高到25m；这张z=19m是有真实标签的验证切片。')
        else:
            parts.append('本例FNO在z=110m这一张的RMSE最低，U-FNO也略优于V2.3；U-Net把减速区预测得过强，PINN则明显低估尾流减速，虽然背景颜色仍接近。V2.3右侧尾流存在条带和局部误差。该建筑已有100～150m共51张1m间隔参考切片，不能用其中一张的排名替代跨高度/建筑总体排名。')
    a=summary['mlp_data']['high_in_range'];b=summary['pinn']['high_in_range']
    parts.extend(['## 必须更正的结论',
        f"上一条回复将PINN未胜过V2.3解释为‘物理损失反而干扰拟合’，证据不足。正确同结构对照为mlp_data：高建筑RMSE从{a['rmse_mps']:.4f}到{b['rmse_mps']:.4f}，约改善{100*(1-b['rmse_mps']/a['rmse_mps']):.2f}%；无量纲连续性RMS从{a['continuity_rms']:.4f}到{b['continuity_rms']:.4f}，约降低{100*(1-b['continuity_rms']/a['continuity_rms']):.1f}%。这说明本轮连续性改善，速度拟合收益很小，不是证实物理损失有害。仅一个种子，不能把微小差异当作统计显著提升。",
        'V2.3具有显式多频位置编码、几何条件调制、边界/尾流分层采样，并专注标量速度；V2.4 MLP是普通Tanh多输出网络，同时预测更多物理量。它们差异很多，不能把相互之间的全部差距归因于PINN。上述差异是已确认实现事实，哪项导致多少收益仍需单独消融验证。',
        'V2.4主要产出是可比较的多模型实验与更完整的扰动/尾流诊断。当前生产模型仍为V2.3；新增模型中U-FNO高建筑RMSE最低，但仍未超越V2.3。增加新指标改善的是解释能力，不意味着原IoU问题或所有局部预测误差已被彻底解决。',
        '这些是既有数据、固定训练配方、单种子的开发性留出实验，不是新几何独立盲测。低23栋只验证了其中2栋；4个高建筑留出只说明已验证组合的表现。尺寸落在单轴范围内也可能远离训练几何分布。',
        '## 文件与查询',
        'V2.3生产查询：在project_v2_3目录执行 `python -m fieldgen.predict --W 16 --L 20 --H 17 --z 19 --output outputs`；高建筑可换成 `--W 24 --L 8 --H 100 --z 110`。生产权重训练过全部28栋，输出不应充当上面的留出验证结果。',
        'V2.4比较查询：在project_v2_4目录执行 `python -m bench.predict --checkpoint artifacts/comparison/ufno/bldg_W24_L8_H100/best.pt --W 24 --L 8 --H 100 --z 110 --output outputs`。默认128网格。展示脚本采用精确原标签视野；CLI用默认几何视野，两者栅格端点略有差别，不应将CLI图像像素直接替代评估用像素。',
        '生产输出可用任意实数z；每1m导出可对z循环，V2.4也有`--all-heights`。多张图共用模型和高度条件，而不是每1m重新训练。',
        '本目录examples.json含单图与全高度原始指标、权重哈希；两个NPZ含本次比较的真实数值和各模型预测；iou_coverage.json记录指标可定义的建筑数。原代码和模型压缩包保持原样，本文为补充说明。',
    ])
    (OUT/'V23_V24_详细说明.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    print('EXAMPLES',[(r['building'],r['z'],r['inlet_mps']) for r in rows])
    for row in rows:
        print(row['building'],{m:row['single_slice'][m]['speed'] for m in METHODS})
    print(OUT/'V23_V24_详细说明.md')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    write_report(examples())
