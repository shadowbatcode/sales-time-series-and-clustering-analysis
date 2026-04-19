# Sales Time Series And Clustering Analysis

该项目面向销售数据的综合挖掘与可视化分析，目标是从时间变化、品类关系和聚类结构三个层面提取业务规律，并形成可用于论文或报告展示的图表和结果文件。

## Project Goals

- 对原始销售数据进行清洗与标准化处理
- 分析单品与品类的时间序列特征
- 构建销售相关性网络
- 对商品或品类进行聚类画像
- 挖掘潜在关联规则与结构特征

## Methods

- 描述统计与异常值检测
- STL 分解与平稳性检验
- 相关性分析与偏相关分析
- KMeans、DBSCAN、层次聚类与高斯混合模型
- PCA 降维
- 关联规则挖掘
- 网络图可视化

## Repository Structure

- `data_preprocess.py`
  预处理、统计分析、聚类和关联规则主流程
- `question1_visualization.py`
  综合图表生成脚本
- `graph.py`
  销售关系网络构建脚本
- 其余 `xlsx`、`png`、`gexf`、`pkl` 文件
  为中间数据、分析结果和图表产物

## Typical Outputs

- 分布分析图
- 时间模式分析图
- 相关性网络图
- 聚类分析图
- 预处理结果表
- 销售图网络文件

## Running

建议先运行 `data_preprocess.py` 生成数据与分析结果，再运行 `question1_visualization.py` 和 `graph.py` 补充展示图表。

## Main Dependencies

- `pandas`
- `numpy`
- `scipy`
- `statsmodels`
- `scikit-learn`
- `mlxtend`
- `matplotlib`
- `seaborn`
- `networkx`
