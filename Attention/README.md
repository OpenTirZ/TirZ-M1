# Multi-Head Attention

This module implements the **Multi-Head Self-Attention (MHSA)** mechanism used in the TirZ M1 Transformer architecture.

Multi-Head Attention enables the model to attend to different parts of the input sequence simultaneously, allowing it to capture complex relationships between tokens.

## Overview

The implementation follows the attention mechanism introduced in the paper:

**"Attention Is All You Need" (Vaswani et al., 2017)**

The module performs:

1. Query, Key, and Value projections
2. Multi-head splitting
3. Scaled Dot-Product Attention
4. Causal masking
5. Attention dropout
6. Head concatenation
7. Output projection

## Architecture

```text
Input Embeddings
       │
       ▼
 ┌─────────────┐
 │ Q K V Linear│
 └─────────────┘
       │
       ▼
 Split Into Heads
       │
       ▼
Scaled Dot Product
     Attention
       │
       ▼
  Causal Mask
       │
       ▼
 Softmax
       │
       ▼
Weighted Sum of Values
       │
       ▼
 Concatenate Heads
       │
       ▼
 Output Projection
       │
       ▼
 Transformer Block
```

## Components

### Query Projection

Projects input embeddings into query vectors.

```text
Q = XWq
```

### Key Projection

Projects input embeddings into key vectors.

```text
K = XWk
```

### Value Projection

Projects input embeddings into value vectors.

```text
V = XWv
```

### Scaled Dot Product Attention

Attention scores are computed using:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √dₖ)V
```

where:

* Q = Queries
* K = Keys
* V = Values
* dₖ = Dimension of each attention head

### Multi-Head Processing

Instead of using a single attention mechanism, the model uses multiple heads.

```text
d_out = num_heads × head_dim
```

Each head learns different relationships within the sequence.

Examples:

* One head may focus on grammar.
* One head may focus on long-range dependencies.
* One head may focus on sentence structure.

## Causal Masking

A causal mask prevents tokens from attending to future positions during training.

Example:

```text
Token 1 → Can see Token 1
Token 2 → Can see Token 1,2
Token 3 → Can see Token 1,2,3
Token 4 → Can see Token 1,2,3,4
```

Future tokens remain hidden.

This ensures autoregressive language modeling behavior.

## Tensor Shapes

Input:

```text
(batch_size, sequence_length, d_in)
```

After projection:

```text
(batch_size, sequence_length, d_out)
```

After head splitting:

```text
(batch_size, num_heads, sequence_length, head_dim)
```

Attention scores:

```text
(batch_size, num_heads, sequence_length, sequence_length)
```

Output:

```text
(batch_size, sequence_length, d_out)
```

## Parameters

| Parameter        | Description                        |
| ---------------- | ---------------------------------- |
| `d_in`           | Input embedding dimension          |
| `d_out`          | Output embedding dimension         |
| `context_length` | Maximum sequence length            |
| `dropout`        | Attention dropout probability      |
| `num_heads`      | Number of attention heads          |
| `qkv_bias`       | Enable bias in Q, K, V projections |

## Features

* Multi-Head Self-Attention
* Causal Masking
* Scaled Dot-Product Attention
* Dropout Regularization
* Output Projection Layer
* GPU Compatible
* PyTorch Native Implementation

## Example Configuration

```python
attention = MultiHeadAttention(
    d_in=768,
    d_out=768,
    context_length=1024,
    dropout=0.1,
    num_heads=12
)
```

## Purpose

This module serves as the core information-routing mechanism of the TirZ M1 Transformer.

By allowing tokens to dynamically attend to relevant context, Multi-Head Attention enables the model to learn linguistic structure, long-range dependencies, semantic relationships, and contextual understanding required for large language model training.
