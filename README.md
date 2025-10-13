# sinusoidal-modular-addition

Minimal, reproducible code for the paper "Provable Benefits of Sinusoidal Activation for Modular Addition."

Implementation of modular addition with **sinusoidal** and **ReLU** activations.  
Includes single length training for both underparameterized and overparameterized regimes (`mlp.py`) and length generalization for out-of-domain regime (`ood.py`), plus SLURM launchers for sweeps (`mlp.sh`, `ood.sh`).

See `env.yml` for base dependencies. Add your W&B API key and entity name to `mlp.py` and `ood.py` to enable metric logging.

## Quickstart 

### Single-length demo
python mlp.py --project "Underparam-demo" --m 3 --p 97 --d 256 --train_size 8000 \
  --seed 1337 --batch_size 1024 --activation sin --learning_rate 1e-3 \
  --weight_decay 0.0 --init_std 0.01 --optimizer muon --epochs 300000

### OOD length-generalization demo
python ood.py --project "OOD-demo" \
  --train_m 2 3 4 5 7 13 19 \
  --test_m 3 7 13 14 38 53 97 201 303 401 512 602 705 811 \
  --p 97 --d 1024 --train_size 8000 --seed 1337 --batch_size 1024 \
  --activation sin --learning_rate 1e-3 --weight_decay 0.0 \
  --init_std 0.01 --optimizer muon --epochs 300000
