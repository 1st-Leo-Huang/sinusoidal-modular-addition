#rms norm of l2 loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
import numpy as np
import random  

def set_seed(seed: int):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed_all(seed)  

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# For PyTorch 2.x:
try:
    torch.set_float32_matmul_precision("highest")
except AttributeError:
    pass

class ModularAdditionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.p + 2
        self.seq_len = 2 * config.m
        self.d = config.d
        
        # Embeddings
        self.tok_emb = nn.Embedding(self.vocab_size, self.d)
        self.pos_emb = nn.Embedding(self.seq_len, self.d)
        
        # LayerNorms
        self.ln1 = nn.LayerNorm(self.d, eps=1e-5)
        self.ln2 = nn.LayerNorm(self.d, eps=1e-5)
        self.ln3 = nn.LayerNorm(self.d, eps=1e-5)
        
        # Attention
        self.W_Q = nn.Linear(self.d, self.d, bias=False)
        self.W_K = nn.Linear(self.d, self.d, bias=False)
        self.W_V = nn.Linear(self.d, self.d, bias=False)
        
        # MLP
        self.mlp1 = nn.Linear(self.d, 4 * self.d)
        self.mlp2 = nn.Linear(4 * self.d, self.d)
        self.activation = self._get_activation(config.activation)
        
        # LM Head
        self.lm_head = nn.Linear(self.d, self.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _get_activation(self, activation_str):
        if activation_str == 'relu':
            return F.relu
        elif activation_str == 'sin':
            return torch.sin
        elif activation_str == 'gelu':
            return F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation_str}")
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    
    def forward(self, input_ids, return_attn=False):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Embed tokens and positions
        tok_emb = self.tok_emb(input_ids)
        pos_emb = self.pos_emb(positions)
        h0 = tok_emb + pos_emb
        
        # Attention block
        h0_norm = self.ln1(h0)
        Q = self.W_Q(h0_norm)
        K = self.W_K(h0_norm)
        V = self.W_V(h0_norm)
        
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d ** 0.5)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).bool()
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Save attention weights if requested
        last_attn = attn_weights[:, -1, :].clone() if return_attn else None
        
        attn_out = torch.matmul(attn_weights, V)
        h1 = h0 + attn_out
        
        # MLP block
        h1_norm = self.ln2(h1)
        mlp_out = self.mlp2(self.activation(self.mlp1(h1_norm)))
        h2 = h1 + mlp_out
        
        # Final LayerNorm and LM head
        h3 = self.ln3(h2)
        logits = self.lm_head(h3)
        
        return (logits, last_attn) if return_attn else logits

def generate_batch(batch_size, m, p):
    summands = torch.randint(0, p, (batch_size, m))
    labels = summands.sum(dim=1) % p
    
    # Build input sequence: [x1, '+', x2, '+', ..., xm, '=']
    input_ids = torch.zeros(batch_size, 2 * m, dtype=torch.long)
    for i in range(m):
        input_ids[:, 2*i] = summands[:, i]
        if i < m - 1:
            input_ids[:, 2*i + 1] = p  # '+' token
        else:
            input_ids[:, 2*i + 1] = p + 1  # '=' token
    
    return input_ids, labels

def evaluate(model, test_inputs, test_labels, batch_size, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = test_inputs.size(0)
    all_attn_weights = []
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            inputs = test_inputs[start_idx:end_idx]
            labels = test_labels[start_idx:end_idx]

            # Note: data is already on the device, no need to move it
            logits, attn_weights = model(inputs, return_attn=True)
            last_logits = logits[:, -1, :]
            
            loss = F.cross_entropy(last_logits, labels)
            preds = last_logits.argmax(dim=-1)
            
            total_loss += loss.item() * inputs.size(0)
            total_correct += (preds == labels).sum().item()
            all_attn_weights.append(attn_weights)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    avg_attn = torch.cat(all_attn_weights, dim=0).mean(dim=0).cpu().numpy()
    
    return avg_loss, accuracy, avg_attn

def main(config):
    run_name = f"m={config['m']}_p={config['p']}_d={config['d']}_act={config['activation']}_{config['optimizer']}_lr={config['lr']}_wd={config['weight_decay']}_bs={config['batch_size']}_ts={config['train_set_size']}_seed={config['seed']}"
    wandb.login(key="")
    wandb.init(project="Transformer", entity="", config=config, name=run_name)
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(config.seed) 

    # Generate fixed training set and move to device
    train_inputs, train_labels = generate_batch(config.train_set_size, config.m, config.p)
    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)

    # Generate fixed test set and move to device
    test_inputs, test_labels = generate_batch(10000, config.m, config.p)
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
    
    # Initialize model
    model = ModularAdditionTransformer(config).to(device)
    
    # Optimizer
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Training loop
    for epoch in range(config.num_epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        
        # 1. Create a random permutation of indices for shuffling
        indices = torch.randperm(config.train_set_size)
        
        # 2. Calculate the number of batches for one full pass over the data
        num_batches = (config.train_set_size + config.batch_size - 1) // config.batch_size
        
        for i in range(num_batches):
            # 3. Get the indices for the current mini-batch
            start_idx = i * config.batch_size
            end_idx = start_idx + config.batch_size
            batch_indices = indices[start_idx:end_idx]
            
            # 4. Slice the data tensors to create the batch
            inputs = train_inputs[batch_indices]
            labels = train_labels[batch_indices]
            
            optimizer.zero_grad()
            logits = model(inputs)
            last_logits = logits[:, -1, :]
            
            loss = F.cross_entropy(last_logits, labels)
            loss.backward()
            
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            
            # Metrics
            preds = last_logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_loss += loss.item() * inputs.size(0)
        
        # Epoch metrics
        train_loss = epoch_loss / config.train_set_size
        train_acc = epoch_correct / config.train_set_size
        epoch_time = time.time() - start_time
        
        # Evaluation
        test_loss, test_acc, avg_attn = evaluate(model, test_inputs, test_labels, config.batch_size, device)
        
        # Log infinity norms
        inf_norms = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                inf_norm = param.data.abs().max().item()
                inf_norms[f"inf_norm/{name}"] = inf_norm
        
        # Log attention weights
        attn_log = {f"attn/pos_{i}": weight for i, weight in enumerate(avg_attn)}
        
        # Wandb logging
        log_data = {
            "epoch": epoch,
            "time/epoch": epoch_time,
            "loss/train": train_loss,
            "loss/test": test_loss,
            "acc/train": train_acc,
            "acc/test": test_acc,
            **inf_norms,
            **attn_log
        }
        wandb.log(log_data)
    
    wandb.finish()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Train a simple Transformer for Modular Addition")

    # Model architecture
    parser.add_argument("--m", type=int, default=3, help="Number of summands")
    parser.add_argument("--p", type=int, default=97, help="Modulus")
    parser.add_argument("--d", type=int, default=388, help="Hidden dimension (embedding size)")
    # This model uses a single activation for the whole network
    parser.add_argument("--activation", type=str, default="sin", help="Activation function to use")

    # Optimizer and training
    # Added 'muon' to the list of choices to support your new command
    parser.add_argument("--optimizer", type=str, default="adamw", choices=['sgd', 'adamw', 'muon'],
                        help="Optimizer to use")
    # Allow both --lr and --learning_rate, but store it as 'lr'
    parser.add_argument("--lr", "--learning_rate", type=float, default=1e-4, dest="lr",
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    # Allow both --epochs and --num_epochs, storing as 'num_epochs' to match the original dictionary key
    parser.add_argument("--num_epochs", type=int, default=100000, dest="num_epochs",
                        help="Number of training epochs")

    # Initialization and other settings
    parser.add_argument("--init_std", type=float, default=0.01, help="Standard deviation for weight initialization")
    # Allow both --grad_clip and --gradient_clipping
    parser.add_argument("--grad_clip", "--gradient_clipping", type=float, default=-1.0, dest="grad_clip",
                        help="Gradient clipping value (-1.0 to disable)")

    # Dataset size
    parser.add_argument("--train_set_size", type=int, default=16384, help="Size of the training dataset")

    parser.add_argument("--seed", type=int, default=1337, help="Random seed")  

    args = parser.parse_args()

    # Convert the parsed arguments (which are in a Namespace object) back to a dictionary.
    # This ensures it's compatible with your existing `main(default_config)` function call.
    config = vars(args)

    main(config)
