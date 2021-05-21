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
source activate /opt/ohpc/pub/apps/tensorflow_2.0.0

# srun -n 1 python3 francesco/src/create_tf_records.py
# srun -n 1 python3 francesco/src/train.py
#srun -n 1 python3 francesco/src/predict.py --ckp_path checkpoints/SEG-ckp_20210521_014649/ --ckp_name test-loss --input_path \
#	"dataset/MRA_P09_denoised/TOF_3D_300um_TR20_TE4p73_sli72_15_biasCor_denoised_5SR.nii.gz"


ckp_folder="checkpoints/"
data_folder="dataset/MRA_P09_denoised/"
for ckp in $(ls "francesco/"$ckp_folder)
do
	for res in $(ls "francesco/"$data_folder)
	do
		python3 francesco/src/predict.py --ckp_path $ckp_folder$ckp"/" --ckp_name test-loss --input_path $data_folder$res
	done
done
