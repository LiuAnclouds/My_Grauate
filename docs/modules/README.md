# 模块说明

项目按数据、特征、模型、实验和结果汇总拆分，训练入口只负责串联流程，不把数据处理逻辑写进模型内部。

| 模块 | 说明 |
| --- | --- |
| `src/dyrift/data_processing/` | 数据预处理和统一格式转换 |
| `src/dyrift/analysis/` | 数据切分、统计分析和训练集合约束 |
| `src/dyrift/features/` | 特征组定义、缓存构建和特征读取 |
| `src/dyrift/models/` | DyRIFT-TGAT、TGAT/GraphSAGE 基线、训练引擎和运行时构建 |
| `src/dyrift/utils/` | 通用工具、指标、CSV 写入和预测文件工具 |
| `src/dyrift/reporting/` | 从训练输出恢复实验结果表 |
| `experiments/common/` | 对比和消融实验的公共 runner |
| `experiments/comparisons/` | Linear Same-Feature、Temporal GraphSAGE、TGAT backbone 对比实验入口 |
| `experiments/ablations/` | DyRIFT-TGAT 模块消融实验入口 |

## 训练链路

```text
data/raw -> analysis -> outputs/features -> runtime -> DyRIFT-TGAT -> outputs/train -> docs/generated CSV
```

`train.py` 是单数据集入口；`experiments/` 下的脚本复用公共 runner，保证不同实验使用同一套数据读取、特征读取和 AUC 汇总逻辑。
