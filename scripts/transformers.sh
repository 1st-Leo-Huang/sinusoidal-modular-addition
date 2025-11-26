#!/bin/bash
#=========================================================================================
# PART 1: SLURM-SBATCH OPTIONS
#=========================================================================================
#SBATCH --job-name=Re-Re-Re
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --output=grokking_%A_%a.out
#SBATCH --error=grokking_%A_%a.err
#SBATCH --array=0-53
#SBATCH --constraint=20g

#=========================================================================================
# YOUR SCRIPT
#=========================================================================================

# --- Setup & Logging ---
INSTALL_DIR="$HOME/software"
MINICONDA_DIR="$INSTALL_DIR/miniconda3"
PROJECT_ENV_NAME="my_project_env"

# --- Sweep values (edit these lists as needed) ---
declare -a activation_values=('relu' 'gelu' 'sin')
declare -a seed_values=(1337 1338 1339)
declare -a train_sizes=(4000 8000 12000 16000 20000 24000)
declare -a weight_decay_values=(0.0)
declare -a optimizer_values=('adamw')

# --- Map SLURM_ARRAY_TASK_ID -> (activation, seed, train_set_size, weight_decay, optimizer) ---
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
na=${#activation_values[@]}
ns=${#seed_values[@]}
nt=${#train_sizes[@]}
nw=${#weight_decay_values[@]}
no=${#optimizer_values[@]}

a_idx=$((  TASK_ID                 % na ))
s_idx=$(( (TASK_ID / na)           % ns ))
t_idx=$(( (TASK_ID / (na * ns))    % nt ))
w_idx=$(( (TASK_ID / (na * ns * nt)) % nw ))
o_idx=$(( (TASK_ID / (na * ns * nt * nw)) % no ))

activation=${activation_values[$a_idx]}
seed=${seed_values[$s_idx]}
train_set_size=${train_sizes[$t_idx]}
weight_decay=${weight_decay_values[$w_idx]}
optimizer=${optimizer_values[$o_idx]}

# --- Print Job Details ---
echo "=========================================================="
echo "Starting on $(hostname)"
echo "Job-ID: ${SLURM_JOB_ID:-N/A}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "activation=$activation | seed=$seed | train_set_size=$train_set_size | weight_decay=$weight_decay | optimizer=$optimizer"
echo "=========================================================="
echo

# --- Conda Initialization and Activation ---
echo "Initializing and activating Conda environment: $PROJECT_ENV_NAME..."
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate "$PROJECT_ENV_NAME" || { echo "Failed to activate project environment"; exit 1; }

# --- Run ---
python /home-nfs/leo/project/Grokking/Final/transformers.py \
    --m 3 \
    --p 97 \
    --d 256 \
    --optimizer "$optimizer" \
    --activation "$activation" \
    --lr 1e-4 \
    --seed "$seed" \
    --weight_decay "$weight_decay" \
    --batch_size 1024 \
    --num_epochs 300000 \
    --init_std 0.01 \
    --train_set_size "$train_set_size" \
    --grad_clip 1.0

echo
echo "=========================================================="
echo "Job finished."
echo "=========================================================="
