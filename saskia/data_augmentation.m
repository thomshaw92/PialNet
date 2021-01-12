%% load data
% load
data_path = '/scratch/cvl/uqsboll2/temp/';
data_folder = 'raw';
seg_folder = 'seg';
save_data_folder = 'aug';
save_seg_folder = 'aug_seg';

verbose = 0;

filenames = dir(fullfile(data_path, data_folder, '*.nii'));
for n = 1:numel(filenames)
    disp(n);
    orig_data = MrImage(fullfile(data_path, data_folder, filenames(n).name));
    orig_seg = MrImage(fullfile(data_path, seg_folder, filenames(n).name));
    % plot and check
    if verbose
        orig_data.plot('z', 27);
        orig_data.plot('z', 27, 'overlayImages', orig_seg, 'overlayMode', 'edge');
    end
    %% downsample
    % downsample
    for nD = 1:orig_data.dimInfo.nDims
        new_samplingPoints{nD} = orig_data.dimInfo.samplingPoints{nD}(1:2:end);
    end
    new_dimInfo = MrDimInfo('samplingPoints', new_samplingPoints);
    down_data = orig_data.resize(new_dimInfo);
    down_seg = orig_seg.resize(new_dimInfo).binarize(0.2);
    % plot and check
    if verbose
        down_data.plot('z', 13, 'displayRange', [1 255]);
        down_data.plot('z', 13, 'overlayImages', down_seg, 'displayRange', [1 255]);
    end
    %% add lower intensity outline
    down_seg_outline = down_seg - down_seg.imerode(strel('sphere', 1), '3d');
    down_data_outline = down_data - down_seg_outline.*down_data.*0.2;
    
    if verbose
        down_data_outline.plot('z', 13, 'displayRange', [1 255]);
    end
    %% add more noise
    noise_image = MrImage(randn(down_data_outline.dimInfo.nSamples).*50);
    noise_data = down_data_outline + noise_image.threshold(0);
    if verbose
        noise_data.plot('z', 13);
        noise_data.plot('z', 13, 'overlayImages', down_seg, 'overlayMode', 'edge');
    end
    
    %% save
    noise_data.parameters.save.path = fullfile(data_path, save_data_folder);
    noise_data.parameters.save.fileName = filenames(n).name;
    noise_data.save();
    
    down_seg.parameters.save.path = fullfile(data_path, save_seg_folder);
    down_seg.parameters.save.fileName = filenames(n).name;
    down_seg.save();
    
    close all;
end
