# 苏州20栋外部验证：V2.3与五种V2.4方法

冻结基线执行 `deliverables/Suzhou20_research_protocol_20261001.md`，使用原始v2.3/v2.4代码与原28栋数据。该基线中新20栋不参与训练、早停或模型选择。新增适配阶段独立保存于 `artifacts/adaptation_v1`，按支持、内层验证、查询整栋划分，不能与原外部直测混称；20栋已被查看，属于回顾性分析。

## 统一入口与当前范围

`research.py` 统一六方法的预测、基线任务和少样本适配入口，仍调用工作区 `project_v2_3` 和 `project_v2_4` 的原实现。网络特征、损失和权重格式保留，代码整合不等于异构网络权重合并。当前新实验只研究跨几何泛化与少样本适配；历史屋顶筛选产物保留。

```powershell
D:/Axianyu/aboutPy312/python.exe project_suzhou_validation/research.py status
D:/Axianyu/aboutPy312/python.exe project_suzhou_validation/research.py baseline report
D:/Axianyu/aboutPy312/python.exe project_suzhou_validation/research.py predict --method v23_mlp --dims 24 8 100 --z 105 --output prediction_new
D:/Axianyu/aboutPy312/python.exe project_suzhou_validation/research.py adapt prepare
D:/Axianyu/aboutPy312/python.exe project_suzhou_validation/research.py adapt run
D:/Axianyu/aboutPy312/python.exe project_suzhou_validation/research.py adapt report
```

预测可选 `v23_mlp, unet_si, unet, fno, ufno, mlp_data`，输出物理坐标/风速NPZ、x/y等比例PNG和检查点来源。适配从总体基线较好的MLP开始，其他五方法保留统一直测基线。进度查看 `artifacts/adaptation_v1/status.json`，新结果查看该目录的 `REPORT.md`。冻结协议及完整当前流程见 `CROSS_GEOMETRY_WORKFLOW.md`。

统一比较清单：V2.3 MLP，以及V2.4 U-Net SI、U-Net、FNO、U-FNO、MLP-data。按用户决定，PINN已从本项目运行队列、结果文件和比较报告中移除。跨版本比较是完整方法对比，不能归因于单一架构因素。

每种方法使用相同原28栋做3个种子的生产训练；另按旋转等价的足迹族分5折开发验证，所有高度跟随建筑族留出。固定训练终点：MLP 4000+300步，V2.4各3000+300步；不同方法计算量不同。六种方法共48个生产/开发任务，复用已完成预测并补算统一分区指标。

运行或续跑：

```powershell
D:/Axianyu/aboutPy312/python.exe C:/Users/lxl/Documents/ChatGPT/new2/project_suzhou_validation/study.py run
```

`artifacts/runner.lock`防止重复启动。V2.4每250步保存恢复状态；MLP未完成任务从该任务头开始。已完成训练和评价会跳过。旧项目数据和模型不覆盖。独立子进程日志位于每个任务的train.log/evaluate.log。

检查当前进度：

```powershell
D:/Axianyu/aboutPy312/python.exe C:/Users/lxl/Documents/ChatGPT/new2/project_suzhou_validation/study.py report
```

主评价覆盖新增低10栋、高10栋、严格H<z<1.5H的全部919个原生数值切片。超出原尺寸范围的13栋也评价并标记外推。所有模型采用同一128×128坐标和有效域，主视野为x/L∈[-2,3]、y/W∈[-1.5,1.5]；它与训练的图像视野不同，属于明确记录的额外分布变化。评价保持原始数值，图像颜色饱和不会截断指标。

按每栋观测高度的Voronoi区间宽度加权，再对建筑等权；0.1m加密层不会简单按张数淹没其他高度。记录屋顶、边缘、侧面、迎风、角流相关几何代理区和尾流误差，候选前10%位置召回及速度损失，以及4/6/8m/s研究阈值下最低观测达标高度和未达标误报。这些阈值不是工程标准。角流代理区是平面四个建筑角点外侧的固定几何邻域；缓存标签只有风速大小，不能由此判断旋涡或分离结构。

模型选择规则只由原建筑分组折外结果拟合，以低/高描述组和z/H段为简单条件。它是待检验选择器，不将H=50解释为物理切换边界。另报告六种方法的固定等权平均和入口剖面基线。配对区间按建筑族重采样，不能按像素或相邻高度充当独立重复。

结果入口：

- `artifacts/REPORT.md`：持续更新的完成进度与已完成种子结果。
- `artifacts/FINAL_REPORT.md`：全体任务完成后生成的完整比较。
- `artifacts/historical/reproduction_audit.json`：旧49个评估复现差异。
- `artifacts/audit.json`：数据来源、20栋实际尺寸与质量状态。
- `artifacts/protocol.json`：冻结方法、随机种子、划分、指标与源代码哈希。
- `artifacts/production/<method>/seed_<seed>/`：每种方法的模型、数值预测和逐栋/逐层指标。
- `artifacts/figures/`：建筑/模型/高度误差热图和曲线。
- `artifacts/gallery/index.html`：20栋、每栋最多5个预定相对高度的六方法图集；坐标单位为米且x/y等比例绘制。
- `artifacts/cfd_sensitivity/REPORT.md`：已有迭代快照与原解、不同快照彼此的场变化；由 `cfd_summary.py` 更新。
- `TRANSFER_ADAPTATION_PLAN.md`：以 V2.3 MLP 作适配起点、优先验证 flexible fine-tuning 的少样本分组方案；注明该收益尚未在这20栋上验证。

独立CFD迭代敏感性：

```powershell
D:/Axianyu/aboutPy312/python.exe C:/Users/lxl/Documents/ChatGPT/new2/project_suzhou_validation/cfd_sensitivity.py
```

初始续算方案曾按最低H、最高H、最长L、高组最大W及一栋高组残差达标案例选择；用户随后要求缩减计算。实际只完成L03的+100/+300/+500快照及U06的+100快照，其余案例跳过。原700次标签保持为主参考。用户已取消网格敏感性；参考场网格离散误差未量化。当前范围见CFD_FOLLOWUP.md及deliverables/研究范围修订_20261001.md。不要直接重启脚本中的原五案例队列。
