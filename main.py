import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from Data import data
from Creating_Data_set import DataSet
from Attention.MultiHeadAttention import MultiHeadAttention
from Activation.GELU import GELU
from LayerNorm.LayerNorm import LayerNorm
from FeedForward.FeedForward import FeedForward
from Transformer.TransformerBlock import TransformerBlock

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

# Functions 
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        probas = torch.softmax(logits, dim=-1) 

        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  

        idx = torch.cat((idx, idx_next), dim=1) 

    return idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def text_to_toekn_ids(text , tokenizer) :
    encoded = tokenizer.encode(text,allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids , tokenizer) :
    decoded_text = tokenizer.decode(token_ids.squeeze().tolist())
    return decoded_text

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Open a file to log training status
    with open('training_log.txt', 'w') as f:
        f.write('Epoch,Step,Train Loss,Val Loss\n') # Write header

        # Main training loop
        for epoch in range(3,num_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() # Calculate loss gradients
                optimizer.step() # Update model weights using loss gradients
                tokens_seen += input_batch.numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    log_message = (f"Ep {epoch+1} (Step {global_step:06d}): "
                                   f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                    print(log_message)
                    # Write training status to file
                    f.write(f"{epoch+1},{global_step},{train_loss:.3f},{val_loss:.3f}\n")

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_toekn_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()


def generate_text_simple_with_top_k(model, idx, max_new_tokens, context_size, top_k=None):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        logits = logits[:, -1, :]

        # Apply Top-K filtering
        if top_k is not None and top_k > 0:
            # Get the top K values and their indices
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # Set all values below the K-th value to -inf
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)

        # Sample from the distribution
        idx_next = torch.multinomial(probas, num_samples=1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

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



class TirZM1(nn.Module):
    def __init__(self , cfg) :
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"] , cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"] , cfg["emb_dim"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.norm_leayer = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"] , cfg["vocab_size"] , bias=False
        )

    def forward(self, x):
        batch_size , seq_length = x.shape

        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_length , device=x.device))

        x = tok_emb + pos_emb

        x = self.drop_emb(x)
        x = self.trf_blocks(x)

        x = self.norm_leayer(x)

        x = self.out_head(x)
        return x


model = TirZM1(TirZ_M1)
toeknizer = tiktoken.get_encoding("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(123)

model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.1)

num_epochs = 13
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, var_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=toeknizer
)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

start_context = "Every effort moves you"
tokenizer_gpt2 = tiktoken.get_encoding("gpt2")

# Generate text with Top-K sampling (e.g., top_k=50)
top_k_token_ids = generate_text_simple_with_top_k(
    model=model,
    idx=text_to_toekn_ids(start_context, tokenizer_gpt2).to(device),
    max_new_tokens=50,
    context_size=TirZ_M1["context_length"],
    top_k=50
)
print(f"\nOutput (Top-K=50): {token_ids_to_text(top_k_token_ids, tokenizer_gpt2)}\n")

# Generate text with a smaller Top-K for more focused output (e.g., top_k=10)
top_k_small_token_ids = generate_text_simple_with_top_k(
    model=model,
    idx=text_to_toekn_ids(start_context, tokenizer_gpt2).to(device),
    max_new_tokens=50,
    context_size=TirZ_M1["context_length"],
    top_k=10
)
print(f"Output (Top-K=10): {token_ids_to_text(top_k_small_token_ids, tokenizer_gpt2)}\n")

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': TirZ_M1,
    'num_epochs': num_epochs,
    'tokens_seen': tokens_seen,
}, 'model.pth')

print("Model, optimizer state, and training parameters saved to model.pth")