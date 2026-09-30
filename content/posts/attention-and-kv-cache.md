---
title: "Scaled Dot-Product Attention 与 KV Cache"
date: 2026-09-30
draft: false
summary: "从缩放点积注意力的公式出发，推导为什么要除以 √d_k，再说明自回归解码里 KV Cache 省了什么、占多少显存。"
categories: ["LLM 基础"]
tags: ["Transformer", "Attention", "KV Cache", "推理优化"]
---

## 1. Scaled Dot-Product Attention

给定查询 $Q \in \mathbb{R}^{n \times d_k}$、键 $K \in \mathbb{R}^{m \times d_k}$、值 $V \in \mathbb{R}^{m \times d_v}$，注意力输出为：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

其中 softmax 按行做，即对每个查询，在所有键上归一化。

### 为什么要除以 $\sqrt{d_k}$

假设 $q, k$ 的各分量独立，均值为 0，方差为 1，则点积

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

的均值为 0，方差为 $d_k$。$d_k$ 较大时，点积的量级约为 $\sqrt{d_k}$，softmax 的输入会落在很大的正负值区间，输出接近 one-hot，梯度几乎为 0。除以 $\sqrt{d_k}$ 把方差拉回 1，softmax 才能保持在梯度正常的区间。

## 2. 自回归解码与 KV Cache

解码第 $t$ 步时，只需要新 token 的查询 $q_t$，而它要和**所有**历史 token 的键、值做注意力：

$$
o_t = \sum_{i=1}^{t} \mathrm{softmax}_i\!\left(\frac{q_t k_i^\top}{\sqrt{d_k}}\right) v_i
$$

历史 token 的 $k_i, v_i$ 在后续步骤里不会变（因果掩码保证它们只依赖更早的 token）。所以可以把它们缓存下来，每一步只为新 token 计算一组 $k_t, v_t$ 并追加。这就是 KV Cache。

| | 不用 KV Cache | 使用 KV Cache |
|---|---|---|
| 每步要算 K/V 的 token 数 | $t$ | $1$ |
| 生成长度为 $T$ 时 K/V 投影的总计算量 | $O(T^2)$ | $O(T)$ |
| 额外开销 | 无 | 显存里保存所有历史 K/V |

注意：每步的注意力本身仍要读取全部 $t$ 个历史 K/V，所以解码阶段通常受**显存带宽**限制，而不是算力。

## 3. 显存占用估算

每个 token 需要缓存的字节数：

$$
\text{bytes per token} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{sizeof(dtype)}
$$

开头的 2 对应 K 和 V。以 Llama-2-7B 为例（32 层，32 个 KV 头，$d_{\text{head}}=128$，fp16）：

$$
2 \times 32 \times 32 \times 128 \times 2 = 524{,}288 \ \text{bytes} = 0.5 \ \text{MiB}
$$

上下文长度 4096 时，单条序列的 KV Cache 约为 $0.5 \times 4096 = 2$ GiB。batch 变大或上下文变长，KV Cache 很快就超过模型权重本身。这也是 MQA / GQA（减少 $n_{\text{kv\_heads}}$）和 KV 量化这类优化的出发点。

## 4. 单头实现示意

```python
import math
import torch

class SingleHeadAttnWithCache(torch.nn.Module):
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.wq = torch.nn.Linear(d_model, d_k, bias=False)
        self.wk = torch.nn.Linear(d_model, d_k, bias=False)
        self.wv = torch.nn.Linear(d_model, d_k, bias=False)
        self.d_k = d_k

    def forward(self, x, cache=None):
        # x: (batch, new_tokens, d_model)
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        if cache is not None:
            k = torch.cat([cache["k"], k], dim=1)
            v = torch.cat([cache["v"], v], dim=1)
        cache = {"k": k, "v": v}

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        # 解码时 new_tokens=1，无需掩码；prefill 阶段需要因果掩码
        if q.size(1) > 1:
            n, m = q.size(1), k.size(1)
            mask = torch.ones(n, m, dtype=torch.bool, device=x.device).tril(m - n)
            scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v, cache
```

`tril(m - n)` 让新 token 能看到全部历史 token，以及它自己之前的新 token。prefill 时 $n = m$，退化为普通下三角掩码。

## 5. 小结

- 除以 $\sqrt{d_k}$ 是为了让点积的方差保持为 1，避免 softmax 饱和。
- KV Cache 用显存换计算，把解码阶段的 K/V 投影从每步 $O(t)$ 降到 $O(1)$。
- KV Cache 的大小与 $n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}}$ 和序列长度成正比，是长上下文推理的主要显存瓶颈。
