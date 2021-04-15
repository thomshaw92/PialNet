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
output_dir_rescaled=/winmounts/uqtshaw/data.cai.uq.edu.au/MRASEG-Q3461/data/synthetic/raw_class-wise_rescaled-fake-histmatch

mkdir -p ${output_dir} ${output_dir_160} ${output_dir_rescaled}
ml ants
ml fsl
# multiply the manual seg by the manual seg mask (seg_TOF_etc) to make seg only
fslmaths ${reference_image_folder}/TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_2SR_resampled_22-52_masked.nii \
-mul ${reference_image_folder}/seg_TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_H400_L300_C10_resized_22-52_masked.nii \
${reference_image_folder}/manual_seg_mult_mask.nii.gz

#first replace the seg values of 1s  with 2s
ImageMath 3 ${reference_image_folder}/seg_inverted.nii.gz ReplaceVoxelValue 1 1 2 ${reference_image_folder}/seg_TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_H400_L300_C10_resized_22-52_masked.nii
#then the seg 0s with ones
ImageMath 3 ${reference_image_folder}/seg_inverted.nii.gz ReplaceVoxelValue 0 0 1 ${reference_image_folder}/seg_inverted.nii.gz
#then all the seg 2s with 0s
ImageMath 3 ${reference_image_folder}/seg_inverted.nii.gz ReplaceVoxelValue 2 2 0 ${reference_image_folder}/seg_inverted.nii.gz
#then multiply the new inverse mask manual seg mask by the original to make brain meat only
fslmaths ${reference_image_folder}/TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_2SR_resampled_22-52_masked.nii \
-mul ${reference_image_folder}/seg_inverted.nii.gz \
${reference_image_folder}/manual_seg_sub_mask.nii.gz

#then, do the same for all of the new synthetic data - multiply them by their masks (at 160 micron and at native)
for x in {1..100} ; do
    fslmaths ${source_image_folder}/${x}.nii.gz -mul ${source_seg_folder}/${x}.nii.gz ${output_dir}/${x}_vessels_only.nii.gz
    echo "vessel ${x}"
done
#and subtract for all of the new synthetic data - mult them by their inverse masks (at 160 micron and at native)
for x in {1..100} ; do
    echo "brain ${x}"
    #invert mask
    ImageMath 3 ${output_dir}/${x}_seg_inverted.nii.gz ReplaceVoxelValue ${source_seg_folder}/${x}.nii.gz 1 1 2
    ImageMath 3 ${output_dir}/${x}_seg_inverted.nii.gz ReplaceVoxelValue ${source_seg_folder}/${x}_seg_inverted.nii.gz 0 0 1 
    ImageMath 3 ${output_dir}/${x}_seg_inverted.nii.gz ReplaceVoxelValue ${source_seg_folder}/${x}_seg_inverted.nii.gz 2 2 0
    fslmaths ${source_image_folder}/${x}.nii.gz -mul ${output_dir}/${x}_seg_inverted.nii.gz ${output_dir}/${x}_brain_only.nii.gz
done
#then histmatch the seg class synthetic to the seg class manual,
for x in {1..100} ; do
    echo "histmatch vessel ${x}"
    ImageMath 3 ${output_dir}/${x}_segmentation_class_histmatched.nii.gz HistogramMatch ${output_dir}/${x}_vessels_only.nii.gz ${reference_image_folder}/manual_seg_mult_mask.nii.gz
done
#and the synth brain to the manual brain
for x in {1..100} ; do
    echo "histmatch brain ${x}"
    ImageMath 3 ${output_dir}/${x}_brain_class_histmatched.nii.gz HistogramMatch ${output_dir}/${x}_brain_only.nii.gz ${reference_image_folder}/manual_seg_sub_mask.nii.gz
done
#and do the poor man's histmatch
for x in {1..100} ; do
    echo "poor man's histmatch $x" 
    #first rescale between 0 and 1
    ImageMath 3 ${output_dir_rescaled}/${x}_brain_only_rescaled.nii.gz HistogramMatch ${output_dir}/${x}_brain_only.nii.gz 0 1
    ImageMath 3 ${output_dir_rescaled/$x}_vessels_only_rescaled.nii.gz HistogramMatch ${output_dir}/${x}_vessels_only.nii.gz 0 1

    #then replace all the values above 0.25 and 0.61 with 0.25 and 0.61
    ImageMath 3 ${output_dir_rescaled}/${x}_brain_only_rescaled.nii.gz ReplaceVoxelValue 0.25 1 0.25 ${source_seg_folder}/${x}_seg_inverted.nii.gz
done
#add the new histmatched classes back together to make new images
for x in {1..100} ; do
    echo "add together ${x}"
    #fslmaths ${output_dir}/${x}_segmentation_class_histmatched.nii.gz -add ${output_dir}/${x}_brain_class_histmatched.nii.gz ${output_dir}/${x}_both_classes_histmatched.nii.gz
    fslmaths ${output_dir_160}/${x}_segmentation_class_histmatched_160.nii.gz -add ${output_dir_160}/${x}_brain_class_histmatched_160.nii.gz ${output_dir_160}/${x}_both_classes_histmatched_160.nii.gz
done
for x in {1..20} ; do
    echo "rescale ${x}"
    #ImageMath 3 ${output_dir}/${x}_both_classes_histmatched.nii.gz RescaleImage ${output_dir}/${x}_both_classes_histmatched.nii.gz 0 1
    ImageMath 3 ${output_dir_160}/${x}_both_classes_histmatched_160.nii.gz RescaleImage ${output_dir_160}/${x}_both_classes_histmatched_160.nii.gz 0 1
done


#160 micron upsampled - will take longer
for x in {1..20} ; do
    fslmaths ${source_image_folder_160}/${x}.nii.gz -mul ${source_seg_folder_160}/${x}.nii.gz ${output_dir_160}/${x}_vessels_only.nii.gz
    #invert mask
    ImageMath 3 ${output_dir_160}/${x}_seg_inverted.nii.gz ReplaceVoxelValue 1 1 2 ${source_seg_folder_160}/${x}.nii.gz
    ImageMath 3 ${output_dir_160}/${x}_seg_inverted.nii.gz ReplaceVoxelValue 0 0 1 ${source_seg_folder_160}/${x}_seg_inverted.nii.gz
    ImageMath 3 ${output_dir_160}/${x}_seg_inverted.nii.gz ReplaceVoxelValue 2 2 0 ${source_seg_folder_160}/${x}_seg_inverted.nii.gz
    fslmaths ${source_image_folder_160}/${x}.nii.gz -mul ${output_dir_160}/${x}_seg_inverted.nii.gz ${output_dir_160}/${x}_brain_only.nii.gz
    ImageMath 3 ${output_dir_160}/${x}_segmentation_class_histmatched_160.nii.gz HistogramMatch ${output_dir_160}/${x}_vessels_only.nii.gz ${reference_image_folder}/manual_seg_mult_mask.nii.gz
    ImageMath 3 ${output_dir_160}/${x}_brain_class_histmatched_160.nii.gz HistogramMatch ${output_dir_160}/${x}_brain_only.nii.gz ${reference_image_folder}/manual_seg_sub_mask.nii.gz
done