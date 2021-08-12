#!/bin/bash
ml ants
#first get size of all slabs together by adding in Z direction
PrintHeader masks_corrected/TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_51_biasCor_mask_c_zipCor.nii
#this will show us the info and the origin etc
#then create new image of 0s with that size 
CreateImage 3 CreateImageBigImagesubP11.nii.gz 72x64x1 0.159375x0.159375x0.16 1090x1277x52 0
#First add the last image on the stack (where the origin begins) to the bigimage.nii.gz
antsApplyTransforms -d 3 -i slab_last.nii.gz -r created_bigimage.nii.gz -n HammingWindowedSinc  -o slab_6_in_correct_space.nii.gz
#if you are doing the masks, use NearestNeighbor for -n

#then go to ITKsnap and manually move the mask images together, starting with the top one on the stack - save the registration transformation.txti
#GO TO TOOLS > REGISTRATION (OR CTRL + R) i
#use manual registration and align the slabs
#add to the next one downi
#after you save the affine with the save icon (for each slab)
#apply the affine to the slabs in antsApplyTransforms
antsApplyTransforms -d 3 -i input_slb_5.nii.gz -r createdBigImage.nii.gz -n HammingWindowedSinc -t transform_slab_5_to_slab_6.txt -o slab_5_in_correct_space.nii.gz
#remember NearestNeighbor for masks
#do all 6
#then add together
ml fsl
fslmaths slab_5_in_correct_space.nii.gz -max slab_6_in_correct_space.nii.gz -max etc output_final_bigimage.nii.gz
