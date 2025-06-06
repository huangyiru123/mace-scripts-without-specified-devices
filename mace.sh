#!/bin/bash
#SBATCH --job-name=mace
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=1
#SBATCH --output=%j.log
#SBATCH --nodelist=gpu1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH -t 15-24:30

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# 启用缓存清理
python -c "import torch; torch.cuda.empty_cache()"

mace_run_train \
    --name "llto_simple_model" \
    --model "MACE" \
    --train_file "combined_sampled_266_structures.extxyz" \
    --valid_file "llto_conservative-valid.extxyz" \
    --test_file "llto_conservative-test.extxyz" \
    --E0s "average" \
    --loss "universal" \
    --energy_weight "10" \
    --forces_weight "800" \
    --energy_key "REF_energy" \
    --forces_key "REF_forces" \
    --eval_interval "1" \
    --error_table "PerAtomMAE" \
    --interaction_first "RealAgnosticDensityInteractionBlock" \
    --interaction "RealAgnosticDensityResidualInteractionBlock" \
    --num_interactions "3" \
    --correlation "3" \
    --max_ell "3" \
    --r_max "5.0" \
    --max_L "1" \
    --num_channels "128" \
    --num_radial_basis "8" \
    --MLP_irreps "16x0e" \
    --scaling "rms_forces_scaling" \
    --lr "0.0005" \
    --weight_decay "1e-8" \
    --ema \
    --ema_decay "0.99" \
    --scheduler_patience "10" \
    --batch_size "1" \
    --valid_batch_size "1" \
    --pair_repulsion \
    --distance_transform "Agnesi" \
    --max_num_epochs "1600" \
    --patience "200" \
    --amsgrad \
    --device "cuda" \
    --clip_grad "10" \
    --restart_latest \
    --default_dtype "float64" \
    --seed "42"
