#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=9:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ey267@cam.ac.uk
#SBATCH -e /nfs/research/birney/users/esther/medaka-img/err/%x-%j.err
#SBATCH -o /nfs/research/birney/users/esther/medaka-img/out/%x-%j.out

module purge
module load cuda/12.2.0

# Initialize Micromamba
MICROMAMBA_PATH=$(which micromamba)
MICROMAMBA_ENV=/hps/software/users/birney/esther/micromamba/envs/indigene-img

# Ensure Micromamba is executable
chmod +x $MICROMAMBA_PATH

# Initialize Micromamba shell
eval "$($MICROMAMBA_PATH shell hook --shell=bash)"

# Activate the environment
micromamba activate $MICROMAMBA_ENV

# Run the pipeline (sweep)
# Create new sweep
# wandb sweep --project vanilla-ae-pytorch-medaka /nfs/research/birney/users/esther/medaka-img/src_files/wandb_yaml/vanilla-ae-hyperparameters.yaml
# # Run sweep agent
CUDA_VISIBLE_DEVICES=2 wandb agent ey267-university-of-cambridge/vanilla-ae-pytorch-medaka/4k46ipqj --count 4
# wandb-slurm --project vanilla-ae-pytorch-medaka --sweep r7t1tmyy \
# start-agents --mem 10GB --cpus-per-task 6 --num-gpus 1 --num-agents 4

# Run the pipeline (single run)
# python3 /nfs/research/birney/users/esther/medaka-img/ml_pipeline/__init__.py --config /nfs/research/birney/users/esther/medaka-img/src_files/wandb_yaml/convnet-ae-v0.yaml --batch_size 32 --epochs 101 --learning_rate 0.0001
