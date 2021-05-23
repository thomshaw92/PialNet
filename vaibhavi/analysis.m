clear;
close all;
clc;

doPlot          = true;
drawManualMask  = true;

addpath('/scratch/cvl/uqvitkya/code/spm12');
addpath(genpath('/scratch/cvl/uqvitkya/code/uniqcforboston/code'));

data_path = '/scratch/cvl/uqvitkya/code/PialNet/vaibhavi/trial_wiener/francesco/PialNet_data/test/raw';
file_name = 'TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_21_biasCor_zipCor_rescaled.nii.gz';


%% Load manual segmentation
manual_data_path = '/afm02/Q3/Q3461/data/MRA_P11/manual_seg/';
manual_raw_file = 'TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_2SR_resampled_22-52_masked.nii';
manual_seg_file = 'seg_TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_H400_L300_C10_resized_22-52_masked.nii';

orig_raw_path = '/afm02/Q3/Q3461/data/MRA_P11/bias_correction_corrected/';
orig_raw_file = 'TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor.nii';

denoised_path = '/afm02/Q3/Q3461/data/MRA_P11/DenoisedImages/SR_5/';
denoised_file = 'TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_5SR.nii.gz';

vessel_seg_path = '/afm02/Q3/Q3461/data/MRA_P11/DenoisedImages/SR_5/vessel_segmentation_intensity/';
vessel_seg_file = 'seg_TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_5SR_VT300_lVT200_VENP10.nii';

mask_path = '/afm02/Q3/Q3461/data/MRA_P11/masks_corrected/';
mask_file = 'TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_mask_c_zipCor.nii';

network_path_300 = '/afm02/Q3/Q3461/results/vaibhavi_trial/francesco_code/epoch42_threshold300/';
%network_file_300 = 'TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_5SR-epoch-42-300.nii.gz';
%network_file_300 = 'TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_2SR-epoch-42-300.nii.gz';
network_file_300 = 'TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_7SR-epoch-42-300.nii.gz';

network_path_400 = '/afm02/Q3/Q3461/results/vaibhavi_trial/francesco_code/epoch42_threshold400/';
network_file_400 = 'TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_5SR-epoch-42-400.nii.gz';
%network_file_400 = 'TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_2SR-epoch-42-400.nii.gz';
%network_file_400 = 'TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_7SR-epoch-42-400.nii.gz';
network_zipcor_400 = 'TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor-epoch-42-400.nii.gz';

manual_raw = MrImage(fullfile(manual_data_path, manual_raw_file));
manual_seg = MrImage(fullfile(manual_data_path, manual_seg_file));
denoised_raw = MrImage(fullfile(denoised_path, denoised_file));
orig_raw = MrImage(fullfile(orig_raw_path, orig_raw_file));
vessel_seg = MrImage(fullfile(vessel_seg_path, vessel_seg_file));
mask = MrImage(fullfile(mask_path, mask_file));
network_300 = MrImage(fullfile(network_path_300, network_file_300));
d_network_400 = MrImage(fullfile(network_path_400, network_file_400));
o_network_400 = MrImage(fullfile(network_path_400, network_zipcor_400));

orig_raw_mask = orig_raw.*mask;
network_300_seg = network_300.*mask;
network_400_seg = d_network_400.*mask;

%% Segmentation comparison

% manual segmentation with denoised underlay
fh1 = manual_raw.maxip('z').plot('rotate90', 1, ...
    'overlayImages', manual_seg.maxip('z'), 'overlayMode', 'edge', ...
    'x', 650:900, 'y', 250:550); fh1.Name = 'manual_raw_manual_seg';

% manual segmentation with original underlay
fh2 = orig_raw_mask.select('z', 1:32).maxip('z').plot('rotate90', 1, ...
    'overlayImages', manual_seg.maxip('z'), 'overlayMode', 'edge', ...
    'x', 650:900, 'y', 250:550); fh2.Name = 'manual';

% threshold segmentation with original underlay
fh3 = orig_raw_mask.select('z', 1:32).maxip('z').plot('rotate90', 1, ...
    'overlayImages', vessel_seg.select('z', 1:32).maxip('z'), 'overlayMode', 'edge', ...
    'x', 650:900, 'y', 250:550); fh3.Name = 'threshold';

% segmentation from the network at 300 threshold
fh4 = orig_raw_mask.select('z', 1:32).maxip('z').plot('rotate90', 1, ...
    'overlayImages', network_300_seg.select('z', 1:32).maxip('z'), 'overlayMode', 'edge', ...
    'x', 650:900, 'y', 250:550); fh4.Name = 'network_300';

% segmentation from the network at 400 threshold
fh5 = orig_raw_mask.select('z', 1:32).maxip('z').plot('rotate90', 1, ...
    'overlayImages', network_400_seg.select('z', 1:32).maxip('z'), 'overlayMode', 'edge', ...
    'x', 650:900, 'y', 250:550); fh5.Name = 'network_400';

%% 

for z = [1,3,7]
    fh_slices{z} = orig_raw_mask.plot('rotate90', 1, ...
        'overlayImages', manual_seg, 'overlayMode', 'edge', ...
        'x', 650:900, 'y', 250:550, 'z', z);
end

%% Evaluating oversegmentation and undersegmentation

% examples of undersegmentation 6,7
% examples of oversegmentation 6
for z = [6,7]
    f1 = orig_raw_mask.plot('rotate90', 1, ...
        'x', 650:900, 'y', 250:550, 'z', z); f1.Name = 'raw';
    f2 = orig_raw_mask.plot('rotate90', 1, ...
        'overlayImages', vessel_seg, 'overlayMode', 'mask', ...
        'x', 650:900, 'y', 250:550, 'z', z); f2.Name = 'threshold';
    f3 = orig_raw_mask.plot('rotate90', 1, ...
        'overlayImages', manual_seg, 'overlayMode', 'mask', ...
        'x', 650:900, 'y', 250:550, 'z', z); f3.Name = 'manual';
    f4 = orig_raw_mask.plot('rotate90', 1, ...
        'overlayImages', network_300_seg, 'overlayMode', 'mask', ...
        'x', 650:900, 'y', 250:550, 'z', z); f4.Name = 'network_300';
    f5 = orig_raw_mask.plot('rotate90', 1, ...
        'overlayImages', network_400_seg, 'overlayMode', 'mask', ...
        'x', 650:900, 'y', 250:550, 'z', z); f5.Name = 'network_400';
end
