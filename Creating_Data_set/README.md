# Dataset Loader

This module contains the dataset and dataloader implementation used for training the **TirZ M1** language model.

## Overview

The `Dataset.py` file converts raw text into tokenized training samples suitable for autoregressive language model training.

The implementation:

* Tokenizes text using the GPT-2 tokenizer (`tiktoken`).
* Creates input-target token pairs using a sliding window approach.
* Supports configurable context length and stride.
* Returns PyTorch `DataLoader` objects for efficient batch training.

## Components

### GPTDatasetV1

A custom PyTorch Dataset that:

1. Encodes the input text into token IDs.
2. Splits tokens into overlapping sequences.
3. Creates:

   * `input_ids` → model input tokens
   * `target_ids` → next-token prediction targets

Example:

Input:

```text
[10, 20, 30, 40]
```

Target:

```text
[20, 30, 40, 50]
```

The model learns to predict the next token at every position.

### create_dataloader()

Creates a PyTorch DataLoader from raw text.

Features:

* Batch generation
* Data shuffling
* Configurable context length
* Sliding window tokenization
* Multi-worker loading support

## Training Strategy

The dataset uses a **sliding window** approach:

```text
Input  : [t1, t2, t3, t4]
Target : [t2, t3, t4, t5]
```

Window movement is controlled by the `stride` parameter.

A smaller stride creates:

* More training samples
* Better data utilization
* Higher memory and training cost

A larger stride creates:

* Fewer training samples
* Faster preprocessing
* Lower overlap between samples

## Parameters

| Parameter     | Description                  |
| ------------- | ---------------------------- |
| `text`        | Raw training text            |
| `batch_size`  | Number of samples per batch  |
| `max_length`  | Context window size          |
| `stride`      | Sliding window step size     |
| `shuffle`     | Shuffle training samples     |
| `drop_last`   | Drop incomplete final batch  |
| `num_workers` | Number of DataLoader workers |

## Dependencies

```bash
pip install torch tiktoken
```

## File Structure

```text
dataset/
│
├── Dataset.py
└── README.md
```

## Purpose

This module is responsible for transforming raw training text into batches of token sequences that can be directly used to train the TirZ M1 transformer model.

It serves as the data preparation pipeline between the raw dataset and the model training process.
