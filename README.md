# 建筑风速场跨几何泛化与少样本适配



## 六种配置

MLP、U-Net SI、U-Net、FNO、U-FNO、MLP-data。MLP作为当前默认配置。不同配置仍各自训练，项目合并不改变网络结构或将异构权重相加。PINN不纳入。

## 统一代码与数据

`engine`集中研究流程；内部`project_v2_3/fieldgen`及`project_v2_4/bench`为已复制进本项目的兼容实现，不再从外部旧项目加载模型代码。内部目录名和`v23_mlp`键仅用于权重及冻结协议兼容。来源哈希记录在`MERGE_MANIFEST.json`。

大型数据和历史权重通过`config.json`及engine/study.py显式引用，未复制、移动或重新生成。V2.5不是脱离本工作区的便携数据包。新实验输出默认位于本项目artifacts。历史脚本快照仅用于追溯，部分仍含历史路径；后续调用统一从v25.py进入，新增专项流程需按对应冻结协议迁移。

原28栋训练与开发；20栋回顾性评价；修订五栋作为未参与本轮学习的外部评价案例。采用共同128×128物理网格、m/s、有效流体掩膜、建筑等权及高度间距加权，严格H<z<1.5H，SSIM范围13 m/s。配置统一不意味着输入信息和计算预算完全相同。

## 入口

在工作区根目录使用具有numpy/scipy/torch等依赖的项目Python：

```powershell
python project_v2_5/v25.py status
python project_v2_5/v25.py predict --method mlp --dims 24 8 100 --z 105 --output project_v2_5/artifacts/query_example
python project_v2_5/v25.py baseline prepare
python project_v2_5/v25.py baseline run
python project_v2_5/v25.py baseline report
python project_v2_5/v25.py adapt prepare
python project_v2_5/v25.py adapt run
python project_v2_5/v25.py adapt report
```

predict/status默认引用已登记的W32修订五栋研究批次检查点，baseline/adapt默认新目录。可用全局`--results`指定另一批次。适配须先有对应基线、协议和评价结果。新批次不自动继承旧实验已完成状态。

## 论文命名

统一写“V2.5框架下的六种模型配置”，图例只使用配置名；既有结果引用时保留实验日期、数据版本、标签及权重来源。更名本身不产生新指标，不构成重新训练或物理验证。

## 合并状态

源码与入口已整合。2026-10-14完成预测迁移核验：六配置各高低两例，共12组，种子20261001；同一权重下新旧入口的风速、坐标、尺寸和高度逐元素一致，最大绝对差为0，来源记录一致。报告见 artifacts/migration_check_20261014/REPORT.md。本次没有重新训练，训练/适配全流程未核验。历史专项分析脚本未全部改造成统一子命令。

