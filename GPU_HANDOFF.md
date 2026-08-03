# GPU 机器交接指南

> 本文档自包含——换设备后无需依赖任何对话记录，照此执行即可完成剩余的 GPU 实验。
> 项目背景与已有结果见 [README.md](README.md)，方案设计见上级目录
> `研究内容二_方案定稿_实现与实验设计.md`。

---

## 1. 环境准备

```bash
git clone <你的仓库地址> && cd mlusd
conda create -n mlusd python=3.11 -y && conda activate mlusd
pip install -e ".[learn]"          # numpy/networkx/pyyaml/scikit-learn + torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q                # 应为 28 passed（无 torch 时 1 skipped）
```

## 2. 数据准备（**必做**，数据不入库）

`data/splits/` 在 `.gitignore` 中，clone 后是空的。二选一：

**方案 A：直接拷贝（推荐，最省事）**
从原机器 `D:\实验\研究二\mlusd\data\splits\` 拷贝以下文件（共约 250 MB）：

| 文件 | 用途 | 必需 |
|---|---|---|
| `d_cal_nbr.pkl.gz` | 正常校准集 24000（含邻域图） | ✅ |
| `d_known_nbr.pkl.gz` | 攻击集 111（六类） | ✅ |
| `d_open.pkl.gz` | 真未知 22（开放集评测） | ✅ |
| `d_cal_nbr_blk_l4_sw.pkl.gz` | 正常集（含跨交易 + L4 + 三明治信号） | 推荐 |
| `d_known_nbr_blk_l4_sw.pkl.gz` | 攻击集（同上，最新完整版） | 推荐 |
| `d_cal_rpc.pkl.gz` / `d_known_rpc.pkl.gz` | 管线对齐配置 B | 可选 |
| `d_phish_ptx.pkl.gz` / `d_benign_ptx.pkl.gz` | PTXPHISH 钓鱼 4986 + 硬负 13067 | 可选 |

**方案 B：从头重建**（需 GCP ADC + Alchemy key，耗时数小时、花 BigQuery 额度）
```bash
gcloud auth application-default login
export ALCHEMY_URL=https://eth-mainnet.g.alchemy.com/v2/<KEY>
python -m experiments.build_dataset dcal --project <GCP项目ID> --ts-lo 2023-08-01 --ts-hi 2023-08-01
# 其余见各 experiments/build_*.py 与 augment_*.py 的 docstring
```

---

## 3. 待做实验一：v1 学习表征（主要 GPU 任务）

### 这是什么

把 **3 个"分布偏离"格子**从统计规则换成学习模型，其余 5 格保持规则版：

| 格 | v0（现状） | v1（待跑） | 参考文献 |
|---|---|---|---|
| L1-j1 | 图统计特征 + IsolationForest | **图自编码器**（GCN 双解码，重构误差） | AnomalyDAE 风格 |
| L2-j1 | DeFi 动作序列 bigram 稀有度 | **微型 Transformer 困惑度** | BERT4ETH 范式 |
| L3-j1 | 调用树 4-gram 稀有度 | **调用树 LM 困惑度** | BlockGPT 缩小版 |

**统一范式**：只在正常数据上训练，异常分 = 重构误差 / 困惑度——与 v0 同量纲，
所以 v0-vs-v1 是干净的消融对比。三个模型都已支持 `device="auto"` 自动用 CUDA。

### 要回答的问题

学习表征能否提升**弱信号类型**？经济类已达 0.97–0.99 没有空间，
目标是 **phishing 0.61 / ponzi 0.55 / sandwich 0.75** 这三类。

### 怎么跑

```bash
# 快速验证脚本能跑（几分钟）
python -m experiments.exp_v1_train --epochs 3 --fit-n 1500 --device cuda

# 正式实验（4090 上预计 30–60 分钟）
python -m experiments.exp_v1_train --epochs 20 --fit-n 12000 --device cuda
```

输出：v0 与 v1 的整体 / 分类型 / D_open 真未知 AUROC 逐项对比（含 Δ）。

### 判读标准

- **弱信号类型（phishing/ponzi/sandwich）提升 ≥ 0.03** → v1 有价值，写入论文作为增强版
- **仅经济类微动、弱类型无提升** → 如实报告"学习表征在本数据规模下未带来增益"，
  论文叙事保持 v0（轻量可解释规则 + 统一校准框架），v1 作为分析章节
- ⚠️ 本地 CPU 小参数烟测过（v0 0.793 / v1 0.795），但严重欠训**不作数**

### 可调超参

`--epochs`（序列模型轮数，建议 20–50）、`--fit-n`（训练用正常样本数，建议 12000–20000）。
模型规模在 `mlusd/signals/nn.py`（`d_model=64, n_layer=2`）与
`mlusd/signals/l1_graph_v1.py`（`hidden=16, emb=8`）中，显存充裕可放大。

---

## 4. 待做实验二：专用模型基线（可选，用于横向对比）

现有基线：IsolationForest / LOF / OneClassSVM / AutoEncoder / RF / HistGradientBoosting
（见 `experiments/exp_baselines.py`、`exp_baselines_deep.py`）。

审稿人可能要求**领域专用模型**对比，候选：

| 模型 | 说明 | 实现难度 |
|---|---|---|
| **SandWatch** | 双任务 GNN 检测三明治（ACM TWEB 2025） | 需复现 |
| **GasTrace** | SVM + 图注意力网络两阶段，96.73% 准确率 | 需复现 |
| **BERT4ETH** | 地址行为序列预训练（有开源） | 中，需适配交易级 |
| **TTAGN / Elliptic++** | 钓鱼/洗钱检测基线 | 中 |

**定位建议**：我们是"一个统一框架覆盖多类型"，不必在单类型上超越专用工具；
论文中应说明"采用文献共识判定条件"（三明治已如此），对比时强调
**统一性 + 开放集 + 可解释 + FP 保证**这些专用工具不具备的能力。

---

## 5. 换设备后如何快速了解现状

1. 读 [README.md](README.md) 的"核心结果"——所有关键数字
2. 读上级目录 `研究内容二_方案定稿_实现与实验设计.md`——完整方案与实验设计
3. `git log --oneline` 看演进（每次提交的 message 都记录了做了什么、效果如何、发现了什么）
4. `experiments/` 下每个脚本的 docstring 都写明了动机与背景

## 6. 重要提醒（踩过的坑）

- **采集管线一致性**：BigQuery 采集含 trace（100%），RPC 无 trace（0%），单凭
  "是否有 trace"即可完美分离（AUROC=1.000）。**跨管线比较必被污染**，
  评测须在同管线内进行（配置 A / 配置 B 各自内部一致）。
- **新增信息源必须同步更新可用性掩码**（`types.py::availability`），
  否则补了数据也会因 `m=0` 被跳过（L4 就踩过这个坑）。
- **BigQuery 必须带 `DATE(block_timestamp)` 分区裁剪**，否则单查扫几百 GB；
  所有方法都有 `dry_run=True` 干跑估算，真跑前先看成本。
- **大查询分批 + 落盘缓存**：一次拉太多易 SSL 中断；原始数据缓存后逻辑修正可离线重算，
  避免重复付费（`data/cache/block_swaps.json.gz` 即此用途）。
- **凭据永不入库**：ADC / API key 走环境变量。
