# Flow Matching with Deterministic Offset and OT-Inspired Pairing

在 [LFM](https://github.com/VinAIResearch/LFM) 基线之上的一组改进，针对 **flow matching 力场的震荡与定性失真**问题。四个改进分散在独立的 branch 中，可以各自单独评估或自由组合。

> Student author: [xiaweijiexox](https://github.com/xiaweijiexox)
> 主要改动集中在 `model/train_flow_latent.py`。

---

## Motivation

Flow matching 相比 diffusion 的一个核心特征是**力场的确定性**——这带来了可控性强、采样速度快的优势，但也意味着一旦目标数据域方差过大,单一力场难以在期望值附近给出结构合理的样本(visual artifact 俗称 "鬼脸")。本仓库的四个 contribution 围绕**两条主线**展开:

1. **给力场一个可学习的确定性偏移**,消除局部震荡 (branch `offset`)
2. **优化 source-target 的配对方式**,让力场学习的分布更紧凑 (branch `conv`, OT-CFM 思想)
3. **优化训练信号的分布**,在更信息密集的时间段多采样 (branch `logit`)

实验表明,这些改动主要在**训练早期的收敛速度**和**最终的定性结果**上有增益。

---

## Contributions

### Branch `offset` — Deterministic offset network

在原始 flow matching 力场 $v_\theta(x, t)$ 之外,额外训练一个小型偏移网络 $\delta_\theta(x, t)$,总力场写成

$$
v_{\text{total}}(x, t) = v_\theta(x, t) + \delta_\theta(x, t).
$$

由于 flow matching 的力场本身是**确定性**的,任何过拟合到局部数据点的"震荡分量"都会反复出现在采样轨迹里。$\delta_\theta$ 提供了一个显式的、可学习的修正通道,让主网络专注于全局结构,把局部调整交给偏移网络。

> Core file: `model/train_flow_latent.py` (branch `offset`)

### Branch `conv` — Convolution-based nearest-neighbor pairing (OT-CFM 简化版)

原始 flow matching 每个 batch 内随机配对 noise ↔ image,导致力场要学的是一个**高方差**的映射。OT-CFM 的思路是通过最优传输把每个 noise 配到距离最近的 image 上,但完整的 OT 需要计算 $B \times B$ 距离矩阵,维度高、开销大。

本 branch 用 **卷积匹配** 近似这一过程:

- 不计算完整距离矩阵,而是对 latent 做局部卷积得到粗粒度特征
- 基于粗粒度特征做最近邻配对,在保留 OT 核心"短距离匹配"直觉的同时,把复杂度压到可接受范围

这样每个 noise 倾向于被送到**最近**的 image,力场要学的映射方差显著减小。

> Core file: `model/train_flow_latent.py` (branch `conv`)

### Branch `logit` — Importance sampling on time

原始 flow matching 训练时 $t \sim \mathrm{Uniform}[0, 1]$,但实际上**中间时间段**(力场曲率最大的区域)对最终生成质量的影响更大。

本 branch 把时间采样换成 **logit-normal** 分布,在 $t \approx 0.5$ 附近做密集采样,两端做稀疏采样。这与 SD3 系列工作里报告的 timestep weighting 思路是一致的——用 importance sampling 把训练信号集中在信息密度最高的区域。

> Core file: `model/train_flow_latent.py` (branch `logit`)

### Branch `conv` (重分配部分)— Noise-to-image redistribution

在 `conv` branch 中,除了卷积匹配本身,还包含训练时的 **noise matrix 重分配**:对每个 batch,把 noise 送到距离最短的 image,模拟 OT-CFM 的效果。这是 `conv` branch 内部的一个子改动,和卷积匹配一起作用。

---

## Results

### Quantitative

| Setting | FID ↓ | 备注 |
|---|---|---|
| Baseline (LFM `celeb_f8_dit`, 475 epochs) | 5.26 | 原始 LFM 报告值 |
| Our reproduction (475 epochs) | 9.24 | 同配置复现 (有偏差,下文解释) |
| Ours + subset + 2500 epochs | 15.9 | **数据集方差控制版,见下** |

> ⚠️ **为什么 FID 反而变高了?** 我们发现在小规模 subset 训练下,FID 相比全量训练是上升的,**但定性结果显著变好**(鬼脸消失)。这是因为 FID 的参考分布与 subset 不匹配,导致度量失真;我们认为这一现象本身值得记录——**低 FID ≠ 生成质量好**,详见下节。

### Qualitative

**2500 epochs, subset,CelebA 256:** 24 张随机采样中,鬼脸数量为 0,仅有 1 张略糊。

**475 epochs, 全量数据,CelebA 256:** 大量样本存在明显的结构变形("一缕扭曲的风吹过"),变形成鬼脸的概率较高。

相关定性分析、Fokker-Planck 视角下的 FM vs DDPM 对比、以及"方差过大 → 鬼脸"的机制推导,见 [technical report](technical-report.pdf)。

---

## Installation

```bash
pip install -r requirements.txt
```

Python `3.10` + PyTorch `1.13.1 / 2.0.0`。

数据集准备:参考 baseline [LFM](https://github.com/VinAIResearch/LFM) 与 [NVAE](https://github.com/NVlabs/NVAE#set-up-file-paths-and-data) 的说明。

---

## Usage

### Training

训练脚本在 `bash_scripts/run.sh`,切换到对应 branch 后运行即可:

```bash
git checkout offset   # or: conv / logit
bash bash_scripts/run.sh
```

### Sampling

```bash
bash bash_scripts/run_test.sh <path_to_arg_file>
```

只需 1 张 GPU。具体参数文件在 `test_args/` 下。

### Evaluation

FID 统计参考 baseline:预计算 stats 放在 `pytorch_fid/` 下,然后运行 `run_test_ddp.sh`。

---

## Branch 组合建议

四个 branch 互相正交,可以自由组合。经验上:

- 只想看最小改动 → `offset` 单独跑
- 想看 OT 风格 → `conv` 单独跑
- 想做 ablation → 分别跑 `offset`, `conv`, `logit`,再跑全部合并
- 实际最强组合:`offset` + `conv` + `logit` + subset 数据集方差控制

---

## Acknowledgement

本仓库 fork 自 [VinAIResearch/LFM](https://github.com/VinAIResearch/LFM),四个 branch 的改动建立在其代码结构之上。感谢原作者开源。