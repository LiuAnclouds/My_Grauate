# 特征构建

特征构建遵循固定流程：原始数据准备、数据分析、特征缓存构建、训练时按配置读取缓存。特征构建阶段不使用验证集标签做后处理，也不把节点 `ID` 当作数值特征输入。

## 输入

本地数据放在 `data/raw/`。Elliptic 和 Elliptic++ 可使用 `src/dyrift/data_processing/scripts/` 下的脚本生成统一的 `gdata.npz` 数据格式；XinYe 使用本地已有的动态图数据格式。

## 流程

```powershell
$env:GRADPROJ_ACTIVE_DATASET="xinye_dgraph"
python src/dyrift/analysis/run_analysis.py
python train.py build_features --outdir outputs/features/<dataset>/<feature_cache>
```

不同数据集切换 `GRADPROJ_ACTIVE_DATASET` 和输出目录。训练时通过本地参数文件指定对应的 feature cache。

## 特征内容

特征由基础节点属性、缺失指示、时间信息、动态图结构统计和历史邻域统计组成。不同数据集原始字段不同，因此最终输入维度可以不同；这不改变模型方法的一致性。

## 缓存约定

```text
outputs/features/<dataset>/<feature_cache>/
```

特征缓存只服务于后续训练，不在公开文档中展开具体训练参数。
