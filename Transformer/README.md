# Transformer Block

![Transformer Block Architecture](image.png)

## Overview

This module implements a **Transformer Block**, the fundamental building block of the TirZ M1 architecture.

A Transformer Block combines:

* Multi-Head Self-Attention
* Feed Forward Network (FFN)
* Layer Normalization
* Residual (Skip) Connections
* Dropout Regularization

By stacking multiple Transformer Blocks, TirZ M1 learns complex contextual relationships and language representations required for next-token prediction.

---

## Architecture

```text
Input
  │
  ▼
LayerNorm
  │
  ▼
Multi-Head Attention
  │
  ▼
Dropout
  │
  ▼
Residual Addition
  │
  ▼
LayerNorm
  │
  ▼
Feed Forward Network
  │
  ▼
Dropout
  │
  ▼
Residual Addition
  │
  ▼
Output
```

This implementation follows the **Pre-LayerNorm (Pre-LN)** Transformer design, where normalization is applied before the Attention and Feed Forward sublayers.

---

## Components

### Multi-Head Self-Attention

The attention layer allows each token to attend to relevant tokens within the context window.

Responsibilities:

* Capture contextual relationships
* Learn token dependencies
* Model long-range interactions
* Build semantic understanding

---

### Feed Forward Network

The FFN processes each token independently after attention has gathered contextual information.

Responsibilities:

* Non-linear transformation
* Feature extraction
* Representation enhancement
* Increased model capacity

---

### Layer Normalization

LayerNorm stabilizes training by normalizing activations across the embedding dimension.

Benefits:

* Faster convergence
* Stable gradients
* Improved optimization
* Better training dynamics

---

### Residual Connections

Residual connections help preserve information across layers.

```text
Output = Sublayer(x) + x
```

Benefits:

* Easier gradient flow
* Stable deep networks
* Reduced vanishing gradients
* Improved training efficiency

---

### Dropout

Dropout is applied after both:

* Multi-Head Attention
* Feed Forward Network

Purpose:

* Reduce overfitting
* Improve generalization
* Encourage robust representations

---

## Implementation

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn = MultiHeadAttention(
            cfg["emb_dim"],
            cfg["emb_dim"],
            cfg["context_length"],
            cfg["drop_rate"],
            cfg["n_heads"],
            cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)

        self.norm_1 = LayerNorm(cfg["emb_dim"])
        self.norm_2 = LayerNorm(cfg["emb_dim"])

        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        shortcut = x
        x = self.norm_1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm_2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
```

---

## Forward Pass

### Step 1: Attention Sublayer

```text
Input
  │
  ▼
LayerNorm
  │
  ▼
Multi-Head Attention
  │
  ▼
Dropout
  │
  ▼
Add Residual
```

The model gathers contextual information from surrounding tokens.

---

### Step 2: Feed Forward Sublayer

```text
Input
  │
  ▼
LayerNorm
  │
  ▼
Feed Forward Network
  │
  ▼
Dropout
  │
  ▼
Add Residual
```

The model transforms contextual representations into richer feature embeddings.

---

## Tensor Shapes

Input:

```text
(batch_size, sequence_length, emb_dim)
```

Example:

```text
(8, 256, 768)
```

After Attention:

```text
(8, 256, 768)
```

After FFN:

```text
(8, 256, 768)
```

Output:

```text
(8, 256, 768)
```

The Transformer Block preserves tensor dimensions.

---

## Why Residual Connections?

Without residual connections:

* Deep networks become difficult to train
* Gradients may vanish
* Information may be lost

Residual paths allow information to flow directly through the network.

Example:

```text
y = F(x) + x
```

where:

* `F(x)` is the learned transformation
* `x` is the shortcut connection

---

## Why Pre-LayerNorm?

TirZ M1 uses the Pre-LN architecture:

```text
LayerNorm
    ↓
Attention / FFN
    ↓
Residual Addition
```

Advantages:

* Improved training stability
* Better gradient propagation
* More reliable scaling to deeper models

Modern LLMs commonly adopt Pre-LayerNorm designs.

---

## Features

* Multi-Head Self-Attention
* Feed Forward Network
* Layer Normalization
* Residual Connections
* Dropout Regularization
* Pre-LN Architecture
* Transformer Compatible
* PyTorch Native Implementation

---

## File Structure

```text
transformer_block/
│
├── TransformerBlock.py
├── image.png
└── README.md
```

---

## Position in TirZ M1

```text
Token Embeddings
       │
       ▼
Positional Embeddings
       │
       ▼
Transformer Block 1
       │
       ▼
Transformer Block 2
       │
       ▼
Transformer Block 3
       │
      ...
       │
       ▼
Transformer Block N
       │
       ▼
Final LayerNorm
       │
       ▼
Output Projection
       │
       ▼
Vocabulary Logits
```

Multiple Transformer Blocks are stacked to form the core of the TirZ M1 language model.

---

## Purpose

The Transformer Block is the primary computational unit of TirZ M1.

It combines attention-based context gathering, non-linear feature transformation, normalization, and residual learning into a single reusable module. By stacking these blocks, the model learns grammar, semantics, reasoning patterns, and long-range dependencies necessary for effective language modeling.
