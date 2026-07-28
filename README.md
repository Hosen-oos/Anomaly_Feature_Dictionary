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

### 类型归因（per-param 对比式字典）

| 配置 | 准确率 |
|---|---|
| 原基线（手写先验 + 聚合格值） | 25% |
| + per-param 参数粒度 | 39% |
| + 对比式权重（k vs 其他攻击） | 45% |
| **+ 利润归因特征** | **62%** |
| 合并 flash_loan + price_manipulation（标注歧义检验） | **75%** |

- **per-param 对比式字典**：权重学"类型 k vs 其他攻击"而非"vs 正常"（各攻击相对正常的偏离
  方向雷同，只有类间对比才有判别力），负权重即否定证据；掩码感知（层不可用不产出证据）。
  把 ponzi/rug_pull 从"永不被预测的死类"救活（0→8、0→5）。
- **利润归因**（`signals/profit.py`）：(地址×代币) 资金流台账 → 10 个与代币小数位无关的比值
  特征（passthrough / drain_imbalance / profit&loss_concentration / victim_is_pool …），
  phishing 归因 0→7。
- **标注歧义发现**：flash_loan 与 price_manipulation 经三次独立尝试均无法分离，但**合并两类后
  准确率 +13 点** → 模型能识别"闪电贷驱动的经济攻击"家族，分不清的是家族内标签划分
  （DeFiHackLabs/DefiLlama 按主要技术归类，而闪电贷驱动的预言机操控同时满足两类定义）。
- **事件解码覆盖**：新增 UniV3 Flash / Balancer / Maker / AaveV1 / ERC3156 闪电贷、Compound
  借贷、dYdX、UniV2 Sync 等签名 → flashloan 动作识别 **20%→75%**，总解码覆盖率 64%。

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
python -m experiments.exp_baselines        # 基线对比（经典 + 深度 + 监督）
python -m experiments.exp_dopen            # 开放集真未知测试
python -m experiments.exp_loto             # LOTO 逐类型留出协议
python -m experiments.exp4_tuning          # 权重更新 + 阈值标定
python -m experiments.exp_contrastive_sweep  # per-param 对比式字典 + λ 扫描 + 标注歧义检验
python -m experiments.exp_profit_check     # 利润归因特征判别力
python -m experiments.exp_decode_coverage  # 事件解码覆盖率
```

### GPU 实验（v1 学习表征）

```bash
pip install -e ".[learn]"
python -m experiments.exp_v1_train --epochs 20 --fit-n 12000 --device cuda
```

v1 把三个格子换成学习模型（L1-j1 图自编码器 / L2-j1 动作序列 Transformer /
L3-j1 调用树 LM，BlockGPT 缩小版），其余 5 格保持 v0 规则版；脚本输出 v0-vs-v1 的
整体、分类型与 D_open 真未知 AUROC 对比。三个模型均支持 `device="auto"` 自动用 CUDA。

## 数据

`data/splits/` 不入库（含 100–200 MB 级 .pkl.gz）。换机器时二选一：拷贝这些文件，
或用 `experiments/build_dataset` + `build_dopen` + `augment_neighborhood` 经 BigQuery 重建
（攻击种子来自研究一 `D:\实验\研究一`）。**凭据（ADC / API key）永不入库。**

## 数据集

| 数据集 | 规模 | 说明 |
|---|---|---|
| `d_cal` / `d_cal_nbr` | 24000 | 正常校准集（_nbr 为邻域图增强版） |
| `d_known` / `d_known_nbr` | 111 | 六类攻击（各 ~20），来自研究一标注种子 |
| `d_known_ext` | 800 | phishing + sandwich 各 400（扩充版） |
| `d_open` | 22 | 真未知（重入/访问控制/业务逻辑，字典中无） |

## 状态

- 系统 v0 全链路 + 参数池 + 学习型聚合器 + per-param 对比式字典 + 利润归因：✅
- 全套实验（基线 / 消融 / 开放集 / LOTO / FP 校准 / 类型归因 / 可扩展性）自洽：✅
- **待做**：v1 学习表征真实训练（GPU，脚本已就绪 `exp_v1_train.py`）、
  价格预言机算 USD 利润、专用模型基线（BERT4ETH / GNN 等）
- **已知边界（诚实报告）**：flash_loan vs price_manipulation 受限于真值标注一致性；
  sandwich 需多笔 bundle 上下文；ponzi 为 2015–16 ETH 交易、无 ERC20 事件

> 设计定稿见上级目录 `研究内容二_方案定稿_实现与实验设计.md`。
