# Layer Normalization

![Layer Normalization](image.png)

## Overview

This module implements **Layer Normalization (LayerNorm)**, a fundamental component of the TirZ M1 Transformer architecture.

Layer Normalization stabilizes training by normalizing activations across the embedding dimension for each token independently. It helps maintain consistent activation distributions throughout the network, enabling faster convergence and more stable gradient flow.

Unlike Batch Normalization, LayerNorm does not depend on batch statistics, making it particularly suitable for Transformer-based language models.

---

## Mathematical Definition

Given an input vector:

```text
x = [x₁, x₂, ..., xₙ]
```

The mean is computed as:

```text
μ = (1/n) Σxᵢ
```

The variance is computed as:

```text
σ² = (1/n) Σ(xᵢ - μ)²
```

The normalized output becomes:

```text
x̂ = (x - μ) / √(σ² + ε)
```

Finally, learnable parameters are applied:

```text
y = γx̂ + β
```

where:

* `γ` (scale) is a learnable parameter
* `β` (shift) is a learnable parameter
* `ε` is a small constant for numerical stability

---

## Implementation

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_out = (x - mean) / torch.sqrt(var + self.eps)

        return norm_out * self.scale + self.shift
```

---

## How It Works

### Step 1: Compute Mean

For each token embedding:

```text
[1.2, 0.8, 2.1, 1.9]
```

Calculate the mean value.

---

### Step 2: Compute Variance

Measure how spread out the embedding values are around the mean.

---

### Step 3: Normalize

Transform values so they have:

```text
Mean = 0
Variance = 1
```

---

### Step 4: Learnable Scaling

Apply trainable parameters:

```text
Output = Normalized × Scale + Shift
```

This allows the model to learn the optimal activation distribution during training.

---

## Why LayerNorm?

Without normalization:

* Activations may grow too large.
* Activations may become too small.
* Training becomes unstable.
* Gradients may explode or vanish.

LayerNorm helps:

* Stabilize training
* Improve convergence speed
* Enable deeper networks
* Maintain healthy gradients
* Improve Transformer performance

---

## LayerNorm vs BatchNorm

| Feature                              | LayerNorm | BatchNorm |
| ------------------------------------ | --------- | --------- |
| Uses Batch Statistics                | No        | Yes       |
| Works with Variable Sequence Lengths | Yes       | Limited   |
| Suitable for Transformers            | Yes       | No        |
| Suitable for NLP                     | Yes       | Rarely    |
| Independent of Batch Size            | Yes       | No        |

For Transformer architectures, LayerNorm has become the standard choice.

---

## Position in Transformer Block

```text
Input
  │
  ▼
Multi-Head Attention
  │
  ▼
Add & LayerNorm
  │
  ▼
Feed Forward Network
  │
  ▼
Add & LayerNorm
  │
  ▼
Output
```

LayerNorm is applied repeatedly throughout the model to maintain stable activations.

---

## Tensor Shapes

Input:

```text
(batch_size, sequence_length, embedding_dimension)
```

Example:

```text
(8, 256, 768)
```

Output:

```text
(8, 256, 768)
```

LayerNorm preserves the original tensor shape.

---

## Learnable Parameters

For an embedding dimension of 768:

```text
Scale Parameters (γ) : 768
Shift Parameters (β) : 768

Total Parameters : 1536
```

These parameters are optimized during training.

---

## File Structure

```text
layer_norm/
│
├── LayerNorm.py
├── image.png
└── README.md
```

---

## Purpose

Layer Normalization is one of the key stabilization techniques used in TirZ M1.

By normalizing token representations before further processing, LayerNorm enables efficient training of deep Transformer networks, improves optimization stability, and helps the model learn richer contextual representations from text data.
