#!/bin/bash
#simple script to histogram match the synthetic data to the validation data BY CLASS
#takes  synthetic and matches to manual seg after separating manual into background, vessel, and brain meat.

#ml singularity
#singularity pull docker://vnmd/ants_2.3.4

#set up the directories
source_seg_folder=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/synthetic/seg
source_image_folder=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/synthetic/raw
source_seg_folder_160=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/synthetic/seg_downsampled
source_image_folder_160=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/synthetic/raw_downsampled
reference_image_folder=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/MRA_P11/manual_seg
output_dir=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/synthetic/raw_class-wise_histmatched
output_dir_160=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/synthetic/raw_class-wise_histmatched_160

mkdir -p ${output_dir} ${output_dir_160}
ml ants
ml fsl
# multiply the manual seg by the manual seg mask (seg_TOF_etc)

fslmaths ${reference_image_folder}/TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_2SR_resampled_22-52_masked.nii \
-mul ${reference_image_folder}/seg_TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_H400_L300_C10_resized_22-52_masked.nii \
${reference_image_folder}/manual_seg_mult_mask.nii.gz

#subtract the manual seg mask by the original
fslmaths ${reference_image_folder}/TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_2SR_resampled_22-52_masked.nii \
-sub ${reference_image_folder}/seg_TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_H400_L300_C10_resized_22-52_masked.nii \
${reference_image_folder}/manual_seg_sub_mask.nii.gz

#then, do the same for all of the new synthetic data - multiply them by their masks (at 160 micron and at native)
for x in {1..100} ; do
    fslmaths ${source_image_folder}/${x}.nii.gz -mul ${source_seg_folder}/${x}.nii.gz ${output_dir}/${x}_vessels_only.nii.gz
    fslmaths ${source_image_folder_160}/${x}.nii.gz -mul ${source_seg_folder_160}/${x}.nii.gz ${output_dir_160}/${x}_vessels_only.nii.gz
done
#and subtract for all of the new synthetic data - sub them by their masks (at 160 micron and at native)
for x in {1..100} ; do
    fslmaths ${source_image_folder}/${x}.nii.gz -sub ${source_seg_folder}/${x}.nii.gz ${output_dir}/${x}_brain_only.nii.gz
    fslmaths ${source_image_folder_160}/${x}.nii.gz -sub ${source_seg_folder_160}/${x}.nii.gz ${output_dir_160}/${x}_brain_only.nii.gz
done
#then histmatch the seg synthetic to the seg manual,
for x in {1..100} ; do
    ImageMath 3 ${output_dir}/${x}_segmentation_class_histmatched.nii.gz HistogramMatch ${output_dir}/${x}_vessels_only.nii.gz ${reference_image_folder}/manual_seg_mult_mask.nii.gz
    ImageMath 3 ${output_dir}/${x}_segmentation_class_histmatched_160.nii.gz HistogramMatch ${output_dir}/${x}_vessels_only.nii.gz ${reference_image_folder}/manual_seg_mult_mask.nii.gz
done
#and the synth brain to the manual brain
for x in {1..100} ; do
    ImageMath 3 ${output_dir}/${x}_brain_class_histmatched.nii.gz HistogramMatch ${output_dir}/${x}_brain_only.nii.gz ${reference_image_folder}/manual_seg_sub_mask.nii.gz
    ImageMath 3 ${output_dir}/${x}_brain_class_histmatched_160.nii.gz HistogramMatch ${output_dir}/${x}_brain_only.nii.gz ${reference_image_folder}/manual_seg_sub_mask.nii.gz
done
#add them back together
for x in {1..100} ; do
    fslmaths ${output_dir}/${x}_segmentation_class_histmatched.nii.gz -add ${output_dir}/${x}_brain_class_histmatched.nii.gz ${output_dir}/${x}_both_classes_histmatched.nii.gz
    fslmaths ${output_dir_160}/${x}_segmentation_class_histmatched_160.nii.gz -add ${output_dir}/${x}_brain_class_histmatched_160.nii.gz ${output_dir}/${x}_both_classes_histmatched_160.nii.gz
done