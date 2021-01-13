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
#source activate /opt/ohpc/pub/apps/tensorflow_2.0.0
source activate /clusterdata/uqfcogno/.conda/envs/TF_2_3_0

#srun -n 1 python3 francesco/src/create_tf_records.py
#srun -n 1 python3 francesco/src/train_seg.py
srun -n 1 python3 francesco/src/predict.py --ckp_path checkpoints/SEG-ckp_20210113_160549/ --input_path PialNet_data/test/raw/imageData.nii --label_path PialNet_data/test/seg/segmentationData.nii
