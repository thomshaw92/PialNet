#!/bin/bash
#SBATCH --job-name=detr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --output="tmp/logs/%j.out"

source /scratch/itee/uqsliu23/miniconda3/bin/activate /scratch/itee/uqsliu23/torch
python3.8 train_seg_count.py