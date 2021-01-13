
task="unlabelled"

if [ "$task" = "labelled" ]; then
  for folders in PialNet_data/bias_correction_corrected/ PialNet_data/bias_correction/
  do
    for filename in $folders*;
    do
      echo $filename
      python3 src/predict.py --ckp_path checkpoints/SEG-ckp_20210113_160549/ --ckp_name epoch-44 --threshold 600 --input_path $filename
      exit 1
    done
  done
elif [ "$task" = "unlabelled" ]; then
  python3 src/predict.py --ckp_path checkpoints/SEG-ckp_20210113_160549/ --ckp_name epoch-44 --threshold 600 --input_path PialNet_data/test/raw/imageData.nii \
          --label_path PialNet_data/test/seg/segmentationData.nii
else
	echo "Error task : "$task
fi
