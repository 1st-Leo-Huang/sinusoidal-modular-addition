#!/bin/bash
#=========================================================================================
# PART 1: SLURM-SBATCH OPTIONS
#=========================================================================================
#SBATCH --job-name=Length-Generalizations-Set-II
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --output=grokking_%A_%a.out
#SBATCH --error=grokking_%A_%a.err
#SBATCH --array=0-209
#SBATCH --constraint=20g

# Exit on error
set -e

#=========================================================================================
# PART 2: CONFIGURATION
#=========================================================================================
# --- Define Parameter Arrays ---

declare -a test_m_values=(
  "3 7 13 14 38 53 97 201 303 401 512 602 705 811"
)

# encode training m-set + p in mp_pairs as "m1,m2,...:p"
declare -a mp_pairs=(
"2,3,4,5,7,13,19:97"
)

declare -a d_values=(1024)
declare -a activation_values=('sin' 'relu')
declare -a seed_values=(1337 1338 1339)

# EXPLICIT train_size sweep
declare -a train_size_values=(4000 8000 16000 32000 64000)

# Weight decay sweep
declare -a weight_decay_values=(0.3 0.1 0.03 0.01 0.003 0.001 0)

# For sine activation ONLY: set to 1 to force WD on both W and V (default is V-only).
FORCE_SINE_WD_BOTH=0

#=========================================================================================
# PART 3: AUTOMATIC JOB ARRAY CALCULATION
#=========================================================================================
num_mp_pairs=${#mp_pairs[@]}
num_d_values=${#d_values[@]}
num_activations=${#activation_values[@]}
num_seeds=${#seed_values[@]}
num_train_sizes=${#train_size_values[@]}
num_weight_decays=${#weight_decay_values[@]}

total_jobs=$((num_mp_pairs * num_d_values * num_activations * num_seeds * num_train_sizes * num_weight_decays))
max_index=$((total_jobs - 1))

echo "Found:"
echo "  - ${num_mp_pairs} (m-set,p) pairs"   
echo "  - ${num_d_values} d_values"
echo "  - ${num_activations} activation_values"
echo "  - ${num_seeds} seeds"
echo "  - ${num_train_sizes} train_size_values"
echo "  - ${num_weight_decays} weight_decay_values"
echo "Total jobs to submit: ${total_jobs}. Array should run from 0 to ${max_index}."
echo

# If the submitted array range exceeds what's needed, skip gracefully.
if (( SLURM_ARRAY_TASK_ID > max_index )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} exceeds max_index=${max_index}. Skipping."
  exit 0
fi

#=========================================================================================
# PART 4: JOB EXECUTION
#=========================================================================================
# Map SLURM_ARRAY_TASK_ID to 6D space:
# (train_size) -> seed -> activation -> weight_decay -> d -> (m-set,p)
train_index=$((SLURM_ARRAY_TASK_ID % num_train_sizes))
tmp=$((SLURM_ARRAY_TASK_ID / num_train_sizes))

seed_index=$((tmp % num_seeds))
tmp=$((tmp / num_seeds))

activation_index=$((tmp % num_activations))
tmp=$((tmp / num_activations))

wd_index=$((tmp % num_weight_decays))
tmp=$((tmp / num_weight_decays))

d_index=$((tmp % num_d_values))
mp_index=$((tmp / num_d_values))

# Actual parameter values
train_size=${train_size_values[$train_index]}
seed=${seed_values[$seed_index]}
activation=${activation_values[$activation_index]}
weight_decay=${weight_decay_values[$wd_index]}
d=${d_values[$d_index]}

# parse "m1,m2,...:p" -> TRAIN_M_ARR[] and p
pair=${mp_pairs[$mp_index]}
IFS=':' read -r m_csv p <<< "$pair"
IFS=',' read -ra TRAIN_M_ARR <<< "$m_csv"

# --- Setup & Logging ---
INSTALL_DIR="$HOME/software"
MINICONDA_DIR="$INSTALL_DIR/miniconda3"
PROJECT_ENV_NAME="my_project_env"

echo "=========================================================="
echo "Starting job: $SLURM_JOB_ID, task: $SLURM_ARRAY_TASK_ID"
echo "Parameters:"
echo "  --p $p"
echo "  --d $d"
echo "  --activation $activation"
echo "  --seed $seed"
echo "  --weight_decay $weight_decay"
echo "  --> train_size = $train_size"
echo "  --> train_m set = ${TRAIN_M_ARR[*]}"           
echo "=========================================================="

# --- Conda Initialization and Activation ---
echo "Initializing and activating Conda environment: $PROJECT_ENV_NAME..."
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate "$PROJECT_ENV_NAME" || { echo "Failed to activate project environment"; exit 1; }

# Optional flag for sine: force WD on both W and V (default is V-only)
sine_wd_flag=""
if [[ "$activation" == "sin" || "$activation" == "sine" ]]; then
  if [[ "$FORCE_SINE_WD_BOTH" == "1" ]]; then
    sine_wd_flag="--sine_wd_both"
  fi
fi

# Choose test_m list (aligned by mp_index)
test_m="${test_m_values[$mp_index]}"
read -ra TEST_M_ARR <<< "$test_m"
test_m_args=()
if (( ${#TEST_M_ARR[@]} > 0 )); then
  test_m_args=(--test_m "${TEST_M_ARR[@]}")
fi

# build train_m args directly from TRAIN_M_ARR
train_m_args=(--train_m "${TRAIN_M_ARR[@]}")

# --- Run Python Script ---
python /home-nfs/leo/project/Grokking/ood.py \
  --project "OOD-ICLR" \
  --p "$p" \
  --d "$d" \
  --train_size "$train_size" \
  --seed "$seed" \
  --batch_size 1024 \
  --optimizer muon \
  --activation "$activation" \
  --learning_rate 1e-3 \
  --weight_decay "$weight_decay" \
  --epochs 300000 \
  --init_std 0.01 \
  --gradient_clipping -1.0 \
  "${train_m_args[@]}" \
  "${test_m_args[@]}" \
  $sine_wd_flag

echo
echo "=========================================================="
echo "Job finished."
echo "=========================================================="
