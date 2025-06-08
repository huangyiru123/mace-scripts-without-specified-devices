#!/bin/bash
#SBATCH --job-name=mace_chno
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32GB  
#SBATCH --cpus-per-task=1
#SBATCH --output=%j.log
#SBATCH --nodelist=gpu1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -t 15-24:30

# 显示SLURM分配的GPU信息
echo "=== SLURM GPU分配信息 ==="
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "原始CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 强制覆盖为GPU 0（在mace_run_train之前重新设置）
echo "=== 强制设置GPU 0 ==="
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=2

echo "强制设置后CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# 显示当前GPU状态
echo "=== 当前GPU状态 ==="
nvidia-smi

# 验证GPU设备设置
echo "=== 验证GPU设备 ==="
python -c "
import torch
import os
print(f'Environment CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"Not set\")}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    for i in range(torch.cuda.device_count()):
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
torch.cuda.empty_cache()
"

echo "=== 开始MACE训练 ==="

# 在命令行中再次确认GPU设置
CUDA_VISIBLE_DEVICES=2 mace_run_train \
    --name "chno_optimized_model" \
    --model "MACE" \
    --train_file "chno_conservative_train.extxyz" \
    --valid_file "chno_conservative_valid.extxyz" \
    --test_file "chno_conservative_test.extxyz" \
    --E0s "average" \
    --loss "universal" \
    --energy_weight "1.0" \
    --forces_weight "100.0" \
    --stress_weight "10.0" \
    --energy_key "REF_energy" \
    --forces_key "REF_forces" \
    --stress_key "REF_stress" \
    --eval_interval "2" \
    --error_table "PerAtomRMSE" \
    --interaction_first "RealAgnosticResidualInteractionBlock" \
    --interaction "RealAgnosticResidualInteractionBlock" \
    --num_interactions "2" \
    --correlation "3" \
    --max_ell "2" \
    --r_max "5.5" \
    --max_L "1" \
    --num_channels "64" \
    --num_radial_basis "10" \
    --MLP_irreps "32x0e" \
    --scaling "rms_forces_scaling" \
    --lr "0.001" \
    --weight_decay "1e-6" \
    --ema \
    --ema_decay "0.99" \
    --scheduler_patience "15" \
    --batch_size "4" \
    --valid_batch_size "8" \
    --pair_repulsion \
    --distance_transform "Agnesi" \
    --max_num_epochs "800" \
    --patience "100" \
    --amsgrad \
    --device "cuda" \
    --clip_grad "10" \
    --restart_latest \
    --default_dtype "float64" \
    --seed "42"