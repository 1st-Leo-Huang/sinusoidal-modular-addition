import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import time
import random
import os
import math
from muon import SingleDeviceMuonWithAuxAdam
from argparse import ArgumentParser
import tempfile  
import shutil    
import atexit    

torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision('highest')
except Exception:
    pass

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Activations: only ReLU and sine -------------------------------------------------
def get_activation(name: str):
    if name in ('sin', 'sine'):
        return torch.sin
    elif name == 'relu':
        return F.relu
    else:
        raise ValueError(f"Unknown activation (allowed: 'relu', 'sin'/'sine'): {name}")
# -------------------------------------------------------------------------------------

# Wrap elements into [-pi, pi)
def wrap_to_pi(t: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * math.pi
    return torch.remainder(t + math.pi, two_pi) - math.pi

class ModularAdditionModel(nn.Module):
    def __init__(self, p, d, m, activation, init_std=0.01):
        super().__init__()
        self.d = d
        self.p = p
        self.m = m
        self.init_std = init_std

        # Only two parameter matrices: W (d x p) and V (p x d)
        self.W = nn.Parameter(torch.empty(self.d, self.p, dtype=torch.float32))
        nn.init.normal_(self.W, mean=0.0, std=self.init_std)

        self.V = nn.Parameter(torch.empty(self.p, self.d, dtype=torch.float32))
        nn.init.normal_(self.V, mean=0.0, std=self.init_std)

        self.activation = activation

    def forward(self, x):
        # x: (batch, m) of token indices in [0, p)
        x_one_hot = F.one_hot(x, num_classes=self.p).to(dtype=torch.float32)   # (batch, m, p)
        x_sum = torch.sum(x_one_hot, dim=1).T                                  # (p, batch)
        hidden = self.activation(self.W @ x_sum)                               # (d, batch)
        logits = (self.V @ hidden).T                                           # (batch, p)
        return logits

def compute_margins(logits, labels):
    """Compute classification margins for each sample in batch."""
    batch_size = logits.size(0)
    correct_logits = logits[torch.arange(batch_size), labels]
    logits_without_correct = logits.clone()
    logits_without_correct[torch.arange(batch_size), labels] = -1e10
    max_other = torch.max(logits_without_correct, dim=1).values
    return (correct_logits - max_other).cpu().numpy()

def compute_norms(model):
    """Compute various weight matrix norms for W and V."""
    norms = {}
    for name, param in model.named_parameters():
        if param.ndim == 2:  # Only for matrices
            matrix_name = name.split('.')[0]  # 'W' or 'V'
            norms[matrix_name] = {
                'l1_inf': torch.max(torch.sum(torch.abs(param), dim=1)).item(),
                'spectral': torch.linalg.matrix_norm(param, 2).item(),
                'inf': torch.max(torch.abs(param)).item(),
                'frobenius': torch.linalg.matrix_norm(param, ord='fro').item(),
            }
    return norms

def main(args):
    set_seed(args.seed)

    # Create a temporary directory for all W&B run files
    _wandb_tmp_dir = tempfile.mkdtemp(prefix="wandb_tmp_")  
    os.environ["WANDB_DIR"] = _wandb_tmp_dir                

    # Robust cleanup on exit (explicit + atexit fallback)
    def _cleanup_wandb_tmp():                               
        try:                                                
            wandb.finish()                                  
        except Exception:                                   
            pass                                            
        shutil.rmtree(_wandb_tmp_dir, ignore_errors=True)   
    atexit.register(_cleanup_wandb_tmp)                     

    # --- wandb --------------------------------------------------------
    wandb.login(key="") # Insert your wandb key here

    act_is_sine = args.activation in ('sin', 'sine')
    v_only = act_is_sine and (not bool(args.sine_wd_both))  # keep WD scoping for optimizer, not for logging

    run_name = (
        f"m={args.m}_p={args.p}_d={args.d}_act={args.activation}_"
        f"{args.optimizer}_lr={args.learning_rate}_wd={args.weight_decay}_"
        f"wdscope={'Vonly' if v_only else 'VW'}_"
        f"bs={args.batch_size}_ts={args.train_size}"
    )
    wandb.init(
        project=args.project,
        entity="", # Insert your entity name here
        name=run_name,
        config=vars(args) | {"wd_scope": "Vonly" if v_only else "VW"},
        dir=_wandb_tmp_dir,     
        save_code=False         
    )
    cfg = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------- Static datasets --------------------------------
    train_size = cfg.train_size
    train_inputs  = torch.randint(0, cfg.p, (train_size, cfg.m), device=device)
    train_targets = torch.sum(train_inputs, dim=1) % cfg.p

    test_inputs   = torch.randint(0, cfg.p, (cfg.test_size, cfg.m), device=device)
    test_targets  = torch.sum(test_inputs, dim=1) % cfg.p
    # -----------------------------------------------------------------

    activation_fn = get_activation(cfg.activation)

    model = ModularAdditionModel(cfg.p, cfg.d, cfg.m, activation_fn, cfg.init_std)\
                .to(device=device, dtype=torch.float32)

    # ---------------- Optimizer & WD scoping --------------------------
    if cfg.optimizer == 'muon':
        if v_only:
            param_groups = [
                dict(params=[model.V], use_muon=True, lr=cfg.learning_rate, weight_decay=cfg.weight_decay),
                dict(params=[model.W], use_muon=True, lr=cfg.learning_rate, weight_decay=0.0),
            ]
        else:
            param_groups = [
                dict(params=[model.V, model.W], use_muon=True, lr=cfg.learning_rate, weight_decay=cfg.weight_decay),
            ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        if v_only:
            param_groups = [
                dict(params=[model.V], lr=cfg.learning_rate, weight_decay=cfg.weight_decay),
                dict(params=[model.W], lr=cfg.learning_rate, weight_decay=0.0),
            ]
        else:
            param_groups = [
                dict(params=[model.V, model.W], lr=cfg.learning_rate, weight_decay=cfg.weight_decay),
            ]
        optimizer = torch.optim.AdamW(param_groups)
    # ------------------------------------------------------------------

    for epoch in range(cfg.epochs):
        tic = time.time()
        model.train()

        # shuffle the *static* dataset
        perm = torch.randperm(train_size, device=device)
        train_inputs_shuf  = train_inputs[perm]
        train_targets_shuf = train_targets[perm]

        epoch_margins, loss_sum, correct = [], 0.0, 0
        num_batches = train_size // cfg.batch_size + (1 if train_size % cfg.batch_size else 0)

        for b in range(num_batches):
            start = b * cfg.batch_size
            end   = min(start + cfg.batch_size, train_size)
            inp   = train_inputs_shuf[start:end]
            tgt   = train_targets_shuf[start:end]

            logits = model(inp)
            loss   = F.cross_entropy(logits, tgt)

            optimizer.zero_grad()
            loss.backward()
            if cfg.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clipping)
            optimizer.step()

            loss_sum += loss.item() * inp.size(0)
            correct  += (logits.argmax(1) == tgt).sum().item()
            with torch.no_grad():
                epoch_margins.append(compute_margins(logits, tgt))

        epoch_time = time.time() - tic
        train_loss = loss_sum / train_size
        train_acc  = correct   / train_size
        min_margin = np.percentile(np.concatenate(epoch_margins), 0.5)

        # -------- Evaluation -----------------------------------------
        model.eval()
        with torch.no_grad():
            test_logits = model(test_inputs)
            test_loss   = F.cross_entropy(test_logits, test_targets).item()
            test_acc    = (test_logits.argmax(1) == test_targets).float().mean().item()

        norms = compute_norms(model)
        generalization_gap = train_acc - test_acc

        # ---- Normalized margin logging policy (updated) ----
        eps = 1e-12
        v_spec = norms['V']['spectral']
        w_fro  = norms['W']['frobenius']

        log_payload = {
            'time/epoch'        : epoch,
            'time/epoch_time_s' : epoch_time,
            'loss/train'        : train_loss,
            'loss/test'         : test_loss,
            'acc/train'         : train_acc,
            'acc/test'          : test_acc,
            'margin/train_min_0.5_percentile': float(min_margin),
            'Generalization Gap': generalization_gap,

            'norm_W/frobenius'  : norms['W']['frobenius'],
            'norm_W/l1_inf'     : norms['W']['l1_inf'],
            'norm_W/spectral'   : norms['W']['spectral'],
            'norm_W/inf'        : norms['W']['inf'],

            'norm_V/frobenius'  : norms['V']['frobenius'],
            'norm_V/l1_inf'     : norms['V']['l1_inf'],
            'norm_V/spectral'   : norms['V']['spectral'],
            'norm_V/inf'        : norms['V']['inf'],
        }

        if act_is_sine:
            with torch.no_grad():
                W_tilde = wrap_to_pi(model.W)
                W_tilde_fro = torch.linalg.matrix_norm(W_tilde, ord='fro').item()

            log_payload.update({
                'margin/normalized_V_2'          : float(min_margin) / (v_spec + eps),
                'margin/normalized_W_F_V_2'      : float(min_margin) / ((v_spec * w_fro) + eps),
                'margin/normalized_Wtilde_F_V_2' : float(min_margin) / ((v_spec * W_tilde_fro) + eps),
                'norm_Wtilde/frobenius'          : W_tilde_fro,
            })
        else:
            # ReLU: only margin / (||V||_2 * ||W||_F)
            log_payload.update({
                'margin/normalized_W_F_V_2'      : float(min_margin) / ((v_spec * w_fro) + eps),
            })
        # ----------------------------------------------------

        wandb.log(log_payload)

    wandb.finish()  

# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--project', type=str, default='MLP-Sep13', help='WandB project name')

    # Model parameters
    parser.add_argument('--seed',  type=int, default=1337)
    parser.add_argument('--m',  type=int, default=3)
    parser.add_argument('--p',  type=int, default=97)
    parser.add_argument('--d',  type=int, default=None)
    parser.add_argument('--init_std', type=float, default=0.01)
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'sin', 'sine'])

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=100)

    # Optimizer parameters
    parser.add_argument('--optimizer', choices=['muon', 'adamw'], default='muon')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--gradient_clipping', type=float, default=-1.0)

    # For sine activations, default WD is V-only; pass this flag to use both W and V.
    parser.add_argument('--sine_wd_both', action='store_true',
                        help="For sine activation: apply weight decay to both W and V (default is V-only). Ignored for ReLU.")

    # list of m values for length generalization testing
    parser.add_argument('--test_m', type=int, nargs='*', default=[],
                        help="Space-separated list of m values to test length generalization. "
                             "Each m uses --test_size samples (default 10,000). Example: --test_m 2 3 5 7")

    args = parser.parse_args()

    if args.d is None:
        args.d = int(np.sqrt(args.p) * 5)
    if args.batch_size is None:
        args.batch_size = int(args.p * args.d * 4)
    if args.train_size is None:
        args.train_size = 4 * args.p * args.d

    if args.learning_rate is None:
        args.learning_rate = 0.01 if args.optimizer == 'muon' else 0.001

    # Normalize/unique-ify test_m for consistency
    args.test_m = sorted(set(int(m) for m in args.test_m))

    main(args)
