#!/bin/bash
#SBATCH -J GTCRN-micro-training       # Job name
#SBATCH --partition=gpu               # Partition name (use "general" or appropriate partition)
#SBATCH --gpus=1
#SBATCH -o training%j.txt             # Standard output file with job ID
#SBATCH -e training%j.err             # Standard error file with job ID
#SBATCH --mail-type=ALL               # Email notifications for all job events
#SBATCH --mail-user=<enter-email>     # Email address for notifications
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=64gb                    # Memory allocation (250 GB)

cd /N/project/icassp2026/GTCRN_micro/GTCRN-Micro || return

# for single GPU
uv run python -m gtcrn_micro.train -C gtcrn_micro/conf/cfg_train_DNS3.yaml -D 0
echo "Running normal training, single GPU"

# for multiple GPUs
# uv run python -m gtcrn_micro.train -C gtcrn_micro/conf/cfg_train_DNS3.yaml -D 0,1,2,3
# echo "Running normal training, mult. GPUs"
