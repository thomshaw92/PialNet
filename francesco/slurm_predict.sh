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

task="unlabelled"

if [ "$task" = "unlabelled" ]; then
  for folders in PialNet_data/bias_correction_corrected/ PialNet_data/bias_correction/
  do
    for filename in $folders*;
    do
      echo $filename
      python3 src/predict.py --ckp_path checkpoints/SEG-ckp_20210113_160549/ --ckp_name epoch-44 --threshold 600 --input_path $filename
    done
  done
elif [ "$task" = "labelled" ]; then
  python3 src/predict.py --ckp_path checkpoints/SEG-ckp_20210113_160549/ --ckp_name epoch-44 --threshold 600 --input_path PialNet_data/test/raw/imageData.nii \
          --label_path PialNet_data/test/seg/segmentationData.nii
else
	echo "Error task : "$task
fi
