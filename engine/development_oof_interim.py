"""Summarize completed whole-building/family OOF folds without touching jobs."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
METHODS = ["v23_mlp", "unet_si", "unet", "fno", "ufno", "mlp_data"]
REGIONS = ["full", "roof", "roof_edge", "windward", "corner_flow_proxy", "side", "wake"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return "—"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.4f} ± {sd:.4f}"


def main():
    protocol = load(ART / "protocol.json")
    progress = load(ART / "progress.json")
    results = {}
    folds = {}
    for method in METHODS:
        results[method] = {}
        folds[method] = []
        for fold in protocol["folds"]:
            path = ART / "development" / method / f"fold_{fold['fold']}" / "evaluation.json"
            if not path.exists():
                continue
            data = load(path)
            folds[method].append(fold["fold"])
            for name, building in data["buildings"].items():
                results[method][name] = building

    complete_methods = [m for m in METHODS if len(folds[m]) == len(protocol["folds"])]
    lines = [
        "# 原28栋开发集整栋/几何族折外结果（阶段汇总）", "",
        f"统一比较六种方法；已完成训练/评价任务：{progress['trained']}/{progress['total_training_jobs']}；完整折外方法：{len(complete_methods)}/6（{', '.join(complete_methods)}）。",
        "每个建筑仅由其未参与训练的折模型预测；同一(W,L)几何族被放在同一折。误差按建筑等权。",
        "表中为建筑间均值 ± 标准差；标准差描述建筑差异，不是均值置信区间。开发集为单一预定种子。",
        "这些指标衡量原28栋范围内的折外泛化；不替代苏州20栋外部评价，后者已被查看，不再称完全盲测。", "",
        "## 整栋折外表现（m/s）", "",
        "| 方法 | 已完成折 | 整场RMSE | 屋顶RMSE | 屋顶边缘RMSE | 迎风RMSE | 侧缘RMSE | 尾流RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        buildings = results[method]
        rows = [buildings[n] for n in sorted(buildings)]
        cells = [method, f"{len(folds[method])}/5"]
        for region in REGIONS:
            cells.append(mean_sd([b["all_heights"]["regions"][region]["rmse_mps"] for b in rows]))
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## 高低组屋顶RMSE（m/s）", "",
              "| 方法 | 低组建筑数 | 低组屋顶RMSE | 高组建筑数 | 高组屋顶RMSE |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        rows = list(results[method].values())
        low = [b for b in rows if b["group"] == "low"]
        high = [b for b in rows if b["group"] == "high"]
        lines.append("| " + " | ".join([
            method, str(len(low)), mean_sd([b["all_heights"]["regions"]["roof"]["rmse_mps"] for b in low]),
            str(len(high)), mean_sd([b["all_heights"]["regions"]["roof"]["rmse_mps"] for b in high]),
        ]) + " |")

    lines += ["", "## 候选位置筛选（前10%屋顶区域）", "",
              "速度损失越低越好，召回率越高越好。它们只相对已保存CFD参考场计算。", "",
              "| 方法 | 低组损失 | 低组召回 | 高组损失 | 高组召回 | 全部损失 | 全部召回 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        rows = list(results[method].values())
        cells = [method]
        for group in ["low", "high", "all"]:
            selected = rows if group == "all" else [b for b in rows if b["group"] == group]
            cells.extend([
                mean_sd([b["all_heights"]["screening"]["top10_regret_mps"] for b in selected]),
                mean_sd([b["all_heights"]["screening"]["top10_recall"] for b in selected]),
            ])
        lines.append("| " + " | ".join(cells) + " |")

    bands = list(next(iter(results["v23_mlp"].values())).get("height_bands", {})) if results["v23_mlp"] else []
    lines += ["", "## 按高度段和区域统计各模型的折外平均RMSE", "",
              "逐栋结果保存在各折evaluation.json；这里按折内未见建筑宏平均，再汇总。小样本分箱只作诊断，不作为稳定模型路由证据。", ""]
    for group in ["low", "high"]:
        for band in bands:
            lines += [f"### {group}组，z/H={band}", "", "| 区域 | " + " | ".join(METHODS) + " |",
                      "|---|" + "---:|" * len(METHODS)]
            for region in ["roof", "roof_edge", "windward", "corner_flow_proxy", "wake"]:
                cells = [region]
                for method in METHODS:
                    vals = [b["height_bands"][band]["regions"].get(region, {}).get("rmse_mps")
                            for b in results[method].values() if b["group"] == group]
                    cells.append(mean_sd(vals))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    lines += ["## 解读边界", "",
              "1. 五折把整栋建筑及相同(W,L)几何族整体留出，避免同一几何的相邻高度切片泄漏到训练与验证两侧。",
              "2. 每个测试建筑只在一个折中出现；按每栋指标等权，不把高度切片当独立样本。报告的跨建筑标准差反映几何异质性。",
              "3. 所有入选方法的折外结果与冻结的苏州20栋外部结果并列报告；位置区域均按冻结掩膜统计。角流区为几何代理区，不能解释成已识别涡旋。",
              "4. 折外相对CFD误差低，仍不等同于现实验证；CFD迭代和网格不确定性需单独报告。", ""]
    out = ART / "development" / "OOF_INTERIM.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}; complete methods={len(complete_methods)}/6; buildings per method="
          + ", ".join(f"{m}:{len(results[m])}" for m in METHODS))


if __name__ == "__main__":
    main()
