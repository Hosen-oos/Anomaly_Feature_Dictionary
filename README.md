# Anomaly Feature Dictionary (mlusd)
### 面向区块链广义异常交易的多层统一信号检测

> 一笔交易进入系统 → 提取 **4 层 × 3 范式**异常签名矩阵 → 按"信息可用性"分组校准 →
> **双通道**判定（已知类型字典匹配 / 未知类型开放识别）→ 输出判定 + 结构化解释。

一个**统一框架**：无需 per-type 训练即可检测多种异常类型，具备开放集检测、可扩展、
per-group 误报保证与结构化可解释性。面向**广义异常交易**（不限于 DeFi）。

---

## 核心结果（真实以太坊数据）

| 无监督检测（攻击 vs 正常，AUROC） | 分数 |
|---|---|
| **★ 本框架（学习聚合 + 量值化）** | **0.790** |
| IsolationForest（扁平特征） | 0.779 |
| LOF / AutoEncoder / OneClassSVM | 0.766 / 0.752 / 0.700 |

- **反超最强无监督基线**，且提供基线没有的 per-group FP 保证 / 可解释 / 类型分类 / 可扩展。
- **开放集**：字典中完全没有的真未知攻击（重入/访问控制等）检测 **AUROC 0.927**，零误塞已知类。
- **消融**：每个设计成分均正贡献（学习聚合 +0.096 / 校准 +0.055 / 量值化 +0.023 / 非稀释 +0.019）。
- **FP 校准**：各可用性组误报率 ≈ 目标 α=1%。
- **信号归因**：经济类攻击 0.92–0.99、phishing/sandwich 弱信号 0.62–0.75、ponzi 稀疏。

## 架构（六模块流水线）

```
tx → M1 采集 → M2 四层×三范式信号矩阵 S + 可用性掩码 m
   → M3 分组 ECDF 校准（共形 p 值）+ 尾部放大
   → { M4 已知类型字典匹配(权重+FP阈值) ; M5 未知开放识别(学习聚合+组内再校准) }
   → M6 优先级判定 + 贡献度解释 → DetectionReport
```

- **四层**：L1 交易图拓扑 / L2 合约交互语义 / L3 EVM 执行轨迹 / L4 链下情报
- **三范式**：分布偏离 / 经济异常 / 信息差异
- **核心主张**：异构可用性——不同异常的信息落在不同层，掩码 + 分组校准如实承接缺层

## 目录

```
mlusd/          M1 采集(collect) · M2 信号(signals) · M3 校准(calibrate) ·
                M4 匹配(match) · M5 开放识别(openset) · M6 决策(decide) · pipeline
                baselines/   基线方法（特征提取 + 对比）
configs/dictionaries/   六类攻击字典 YAML
experiments/    数据构建 / 评测 / 消融 / 基线 / 开放集 脚本
tests/          单测 + 合成数据端到端（28 项）
data/splits/    数据集（gitignore，不入库；换机器需重建或拷贝）
```

## 环境

```bash
pip install -e .                    # numpy / networkx / pyyaml / scikit-learn
pip install -e ".[learn]"           # torch（v1 学习模型 / 深度基线，建议 GPU 机器）
pip install google-cloud-bigquery   # 数据采集
gcloud auth application-default login   # BigQuery ADC（换机器各配一次）
```

采集脚本用 `--project <你的 GCP 项目 ID>` 传入项目。

## 跑实验

```bash
python -m pytest -q                        # 28 项测试
python -m experiments.demo                 # 合成数据端到端演示
python -m experiments.exp_refresh_all      # 实验二/三：开放集 AUROC + 消融
python -m experiments.exp_baselines        # 基线对比
python -m experiments.exp_dopen            # 开放集真未知测试
python -m experiments.exp4_tuning          # 权重更新 + 阈值标定
```

## 数据

`data/splits/` 不入库（含 100–200 MB 级 .pkl.gz）。换机器时二选一：拷贝这些文件，
或用 `experiments/build_dataset` + `build_dopen` + `augment_neighborhood` 经 BigQuery 重建
（攻击种子来自研究一 `D:\实验\研究一`）。**凭据（ADC / API key）永不入库。**

## 状态

- 系统 v0 全链路 + 参数池 + 学习型聚合器 + 真实数据管线：✅
- 全套实验（基线/消融/开放集/FP 校准）自洽：✅
- 待做：v1 学习表征真实训练（GPU）、价格预言机算 USD 利润、专用模型基线（GNN 等）

> 设计定稿见上级目录 `研究内容二_方案定稿_实现与实验设计.md`。
