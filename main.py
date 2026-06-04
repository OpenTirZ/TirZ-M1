import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt

from Data import data
from Creating_Data_set import DataSet
from Attention.MultiHeadAttention import MultiHeadAttention
from Activation.GELU import GELU

torch.manual_seed(123)

# Model Info 
TirZ_M1 = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False,
    "model_type" : "gpt"
}

# Getting Data 
text_data = data.gettingData()

# Split the Data into training and validation 
train_ratio = 0.90
split_ids = int(train_ratio * len(text_data))

train_data = text_data[:split_ids]
test_data = text_data[split_ids:]

# Creating DataSet for both training and vlidation
train_loader = DataSet.create_dataloader(
    train_data ,
    batch_size = 2,
    max_length = TirZ_M1["context_length"],
    stride = TirZ_M1["context_length"],
    shuffle = True,
    drop_last = True,
    num_workers = 0
)

var_loader = DataSet.create_dataloader(
    test_data ,
    batch_size = 2,
    max_length = TirZ_M1["context_length"],
    stride = TirZ_M1["context_length"],
    drop_last = False ,
    shuffle=False ,
    num_workers = 0
)

