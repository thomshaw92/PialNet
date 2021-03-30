#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=PialNet
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=50000
#SBATCH -o out_wiener.txt
#SBATCH -e error_wiener.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1

module load cuda/10.0.130
module load gnu7
module load openmpi3
module load anaconda/3.6

#conda create -y --name lab2im_env python=3.6
conda env list

source activate lab2im_env
# srun -n 1 python3 francesco/src/train.py
srun -n 1 python3 francesco/src/predict.py --ckp_path checkpoints/SEG-ckp_20210328_002027/ --ckp_name test-loss --input_path dataset/original/test/raw/TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_2SR_resampled_22-52_masked.nii --label_path dataset/original/test/seg/seg_TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_H400_L300_C10_resized_22-52_masked.nii
conda deactivate
