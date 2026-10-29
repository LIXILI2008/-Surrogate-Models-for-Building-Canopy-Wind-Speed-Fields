"""Freeze new geometries and baseline checkpoints before receiving CFD labels."""
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from study import ART,OLD24,EXTERNAL,METHODS,SEEDS,read,write,digest
OUT=ART/'prospective_five_v1'
CASES=[('P01',[18,18,18],'低建筑范围内插值；检验原模型是否保持已学规律'),('P02',[26,14,90],'高建筑范围内插值；与低组插值对照'),('P03',[18,14,140],'细高组合；48栋各尺寸轴内但凸包外，检验联合几何外推'),('P04',[60,100,24],'宽长低体块；宽度超过48栋最大值约8.1%，检验宽度外推'),('P05',[32,44,250],'高体块；高度超过48栋最大值约8.7%，检验高度外推')]

def classify(d,a):
    lo=a.min(0);hi=a.max(0);scale=hi-lo
    return dict(axis_inside=bool(np.all(d>=lo)&np.all(d<=hi)),hull_inside=bool(Delaunay((a-lo)/scale).find_simplex((d-lo)/scale)>=0))

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Prospective protocol already frozen')
    old=read(OLD24/'manifest.json');ext=read(EXTERNAL/'manifest.json');a=np.array(old['dims']);b=np.r_[a,np.array([r['dims'] for r in ext['geometry'].values()])]
    cases=[]
    for label,values,reason in CASES:
        d=np.array(values);w,l,h=values
        assert not any(np.allclose(sorted(d[:2]),sorted(t[:2]),atol=.11) for t in b)
        zs=sorted(set([float(z) for z in range(h+1,int(np.ceil(1.5*h))) if z<1.5*h]+[h+i/10 for i in range(1,11)]))
        cases.append(dict(id=label,W=w,L=l,H=h,directory=f'bldg_W{w}_L{l}_H{h}',reason=reason,old28=classify(d,a),all48=classify(d,b),evaluation_bounds_xy=[-2*l,-1.5*w,3*l,1.5*w],evaluation_z_m=zs,ratio_bounds=[1,1.5]))
    hashes={f'{m}/{s}':digest(ART/f'production/{m}/seed_{s}/best.pt') for m in METHODS for s in SEEDS}
    p=dict(created='2026-10-03',state='GEOMETRY_AND_ZERO_SHOT_PROTOCOL_FROZEN',cases=cases,
        original28_bounds_WLH=[a.min(0).tolist(),a.max(0).tolist()],all48_bounds_WLH=[b.min(0).tolist(),b.max(0).tolist()],
        fixed_inlet='Wind +x; U(z)=3*(z/10)^0.4 m/s. Match existing turbulence, roughness and ground conventions; verify actual case settings.',
        meaning='L is streamwise x length; W is crosswind y width; H vertical. Origin at footprint centre, ground z=0. Listed view is evaluation crop, NOT CFD domain.',
        evaluation='Six original methods x three frozen seeds. Strict H<z<1.5H; common128 grid, existing height-integrated building-macro metrics; per-building and region results. No training or tuning on these five.',
        checkpoints_sha256=hashes,adaptation_extension='Not yet registered: choose and freeze any additional adapted checkpoint using OLD data before opening the five new labels; otherwise report adapted-model evaluation as exploratory.',
        quality='Save case/data, residual and monitor history, inlet/boundary details, mesh stats, actual dimensions and timings. Keep iteration700 snapshot for comparability if produced; record whether criteria pass. Do not label unconverged result validated truth.',
        holdout='Do not inspect these labels to change architecture, supports, hyperparameters, masks or selection. Report all five including failures. Five geometries provide limited confirmation, not population-wide proof.')
    write(OUT/'protocol.json',p)
    lines=['# 新增5个前瞻评价体块','','尺寸顺序请注意：表格为长L、宽W、高H；文件夹为W/L/H。L沿+x来流，W沿y。','', '|编号|长L/m|宽W/m|高H/m|目录|选择原因|','|---|---:|---:|---:|---|---|']
    for r in cases:lines.append(f"|{r['id']}|{r['L']}|{r['W']}|{r['H']}|{r['directory']}|{r['reason']}|")
    lines+=['','五个均无重复平面族（含旋转等价检查）；P01、P02在原28栋三维几何凸包内。P03虽然各尺寸落在48栋单轴范围内，但组合处在其凸包外。P04/P05分别越出现有48栋宽度/高度范围。数学凸包用于描述数据覆盖，不保证物理泛化。','', '## 模拟与交付','', '1. 使用与既有案例一致的固定来流、湍流和地面条件；实际设置需从case核验，不能只复制文件夹名。坐标原点为建筑投影中心，地面z=0。','2. CFD计算域按建筑尺度和现有经核验的设置建立。protocol中的evaluation_bounds_xy只是出图评价视野，不可直接当求解域。P05比已有最高体块高20m，其计算域和网格成本可能更大。','3. 保留case/data、残差、代表位置风速或流量监测、网格信息和实际尺寸。700次快照可用于旧协议对照，但迭代数不能替代收敛/稳定性证据。新一轮没有要求你全面重做网格敏感性。','4. 主研究只需要严格H<z<1.5H。protocol.json逐栋列出1m主切片及屋顶以上1m内0.1m加密切片；H和1.5H不进入主指标。无需为本轮新增评价制作全部楼下切片。','5. 导出数值坐标及风速大小，条件允许一并保存u/v/w；图片仅作展示。统一128×128重采样在后处理中完成，避免直接从伪彩色图片反解标签。','6. 用timing_template.tsv记录建模、网格、求解、导出分别耗时及设备。时间留空表示未知；不要填猜测值。','', '## 冻结范围','', '五栋几何、评价切片、六原模型18个检查点哈希已冻结。五栋不加入训练或调参。新增适配模型如需参加确认，必须在查看五栋CFD标签之前，依据旧数据另行锁定检查点；此项尚未登记。','它们是有目的设计的独立体块，不宣称是苏州某五栋实楼，也不宣称在所有可能候选中统计最优。']
    (OUT/'给用户的5体块模拟清单.md').write_text('\n'.join(lines),encoding='utf-8')
    headers=['id','modeling_minutes','meshing_minutes','solver_wall_minutes','export_minutes','cpu_model','cpu_cores_used','gpu_model_if_used','cells','iterations','quality_pass','notes']
    (OUT/'timing_template.tsv').write_text('\t'.join(headers)+'\n'+'\n'.join(r['id']+'\t'*(len(headers)-1) for r in cases)+'\n',encoding='utf-8')
    print(OUT/'给用户的5体块模拟清单.md')

if __name__=='__main__':main()
