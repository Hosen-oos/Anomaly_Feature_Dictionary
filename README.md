# mlusd — 多层统一信号检测（研究内容二）

面向区块链**广义异常交易**的检测框架：4 层 × 3 范式异常签名矩阵 + 双通道（已知类型
字典匹配 / 未知类型开放识别）。设计定稿见 `../研究内容二_方案定稿_实现与实验设计.md`。

## 环境

```bash
pip install -e .            # numpy / networkx / pyyaml / scikit-learn
pip install -e ".[learn]"   # 额外装 torch / torch-geometric（仅 v1 学习模型需要，建议 GPU 机器）
pip install google-cloud-bigquery   # 数据采集需要
gcloud auth application-default login   # BigQuery ADC（换机器各配一次）
```

GCP 项目 ID 在采集脚本里传 `--project`。当前用 `project-b471d110-9146-4221-872`。

## 目录

```
mlusd/          M1 采集(collect) · M2 信号(signals) · M3 校准(calibrate) ·
                M4 匹配(match) · M5 开放识别(openset) · M6 决策(decide) · pipeline
configs/dictionaries/   六类攻击字典 YAML
experiments/    build_dataset / augment_neighborhood / eval_v0 / eval_b / exp4_tuning / demo
tests/          单测 + 合成数据端到端（28 项）
data/splits/    数据集（gitignore，不入库；换机器需重建或拷贝）
```

## 换机器（→ 4090 跑 GPU 实验）

`data/splits/` 不入库。到新机器后二选一：

1. **拷贝**（推荐，省额度）：把 `data/splits/*.pkl.gz` 从原机器拷过来。
2. **重建**（BigQuery，约 300GB 免费额度内）：
   ```bash
   # 正常校准集 + 攻击集（种子在 D:\实验\研究一，需同步）
   python -m experiments.build_dataset dcal   --project <PID> --ts-lo 2023-08-01 --ts-hi 2023-08-01
   # D_known 由研究一种子构建（见 dataset/seeds.py + build_dknown_from_seeds）
   python -m experiments.augment_neighborhood dknown   # 邻域增强（可选）
   python -m experiments.augment_neighborhood dcal
   ```

## 跑实验

```bash
python -m pytest -q                    # 28 项测试
python -m experiments.demo             # 合成数据端到端演示
python -m experiments.eval_b nbr       # 真实数据检测评测（邻域增强版）
python -m experiments.exp4_tuning      # 实验四：权重更新 + 阈值标定
```

**GPU 实验（4090 机器）**：v1 学习模型 `signals/{l1_graph_v1,l2_semantic_v1,l3_trace_v1}.py`，
用 `factory.v1_extractors()`。现仅合成数据测通，待接真实 D_cal 训练（v0-vs-v1 消融）。

## 关键实证结论（真实数据，截至 2026-07）

- 正常误报率 ≈ 目标 α=1%（校准机制在真实分布上有效）
- 实验四：阈值标定方式 F1 vs FP控制 → 正常误判 KNOWN 25%→1.4%；KNOWN 分类 0→激活
- 信号归因：经济类(flash_loan/price_manip)靠 L2 语义；phishing 靠 L4，非 L1 图扇入
