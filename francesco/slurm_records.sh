#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=PialNet
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=300000
#SBATCH -o out_wiener.txt
#SBATCH -e error_wiener.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cuda/10.0.130
module load gnu7
module load openmpi3
module load anaconda/3.6

#source activate /opt/ohpc/pub/apps/tensorflow_2.0.0
source activate lab2im_env

srun -n 1 python3 francesco/src/create_tf_records.py
conda deactivate
