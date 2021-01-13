#!/bin/bash
#simple script to histogram match the synthetic data to the validation data
#takes random sample of 20 synthetic and matches to each of the validation samples (n=13)

#ml singularity
#singularity pull docker://vnmd/ants_2.3.4

#set up the directories

source_image_folder=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/validation_data_sensible_naming/aug
reference_image_folder=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/validation_data_sensible_naming
output_dir=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/validation_data_sensible_naming/histmatched_aug
seg_dir_input=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/validation_data_sensible_naming/aug_seg
seg_dir_output=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/validation_data_sensible_naming/histmatched_aug_seg
#run the command
count=1
shopt -s lastpipe
for x in {1..13} ; do 
	cd ${source_image_folder}
	ls |sort -R |tail -20 |while read file; do
		echo "Validation sample ${x} with source image ${file} goes to output: ${count}.nii.gz" >> ${output_dir}/histmatch_log.txt 
		singularity exec ${output_dir}/ants_2.3.4_latest.sif ImageMath 3 ${output_dir}/augmented_histmatched_${count}.nii.gz HistogramMatch ${source_image_folder}/${file} ${reference_image_folder}/${x}.nii.gz 
		ls ${seg_dir_input}/${file}
		cp ${seg_dir_input}/${file} ${seg_dir_output}/${count}.nii.gz
		echo ${count}
	let "count++"
	done
done
