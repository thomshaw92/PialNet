#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=PialNet
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=50000
#SBATCH -o out_wiener.txt
#SBATCH -e error_wiener.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cuda/10.0.130
module load gnu7
module load openmpi3
module load anaconda/3.6

#conda create -y --name lab2im_env python=3.6
#conda env list
#source activate lab2im_env
srun -n 1 python3 francesco/src/train.py

#input_path="dataset/MRA_P09_denoised/TOF_3D_300um_TR20_TE4p73_sli72_15_biasCor_denoised_2SR.nii.gz"
#srun -n 1 python3 francesco/src/predict.py --ckp_path checkpoints/SEG-ckp_20210328_002027/ --ckp_name test-loss --input_path $input_path
#conda deactivate
