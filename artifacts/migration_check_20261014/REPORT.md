# V2.5入口迁移一致性核验

同一历史权重、种子20261001，共六配置×高低两个案例，分别调用原research.main与V2.5命令行。原入口仅绑定相同结果目录；未更改推理逻辑。

|配置|案例|风速最大绝对差m/s|逐元素一致|权重相同|
|---|---|---:|---|---|
|mlp|low|0.0|True|True|
|mlp|high|0.0|True|True|
|unet_si|low|0.0|True|True|
|unet_si|high|0.0|True|True|
|unet|low|0.0|True|True|
|unet|high|0.0|True|True|
|fno|low|0.0|True|True|
|fno|high|0.0|True|True|
|ufno|low|0.0|True|True|
|ufno|high|0.0|True|True|
|mlp_data|low|0.0|True|True|
|mlp_data|high|0.0|True|True|

范围：预测入口及现有权重兼容；没有重新训练，没有验证训练/适配全流程，没有重新认证原指标或CFD。
