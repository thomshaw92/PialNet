%% load data
% load
orig_data = MrImage('/afm02/Q3/Q3503/synthetic/aug/data_1.nii');
orig_seg = MrImage('/afm02/Q3/Q3503/synthetic/aug/seg_1.nii');
% plot and check
orig_data.plot('z', 27);
orig_data.plot('z', 27, 'overlayImages', orig_seg, 'overlayMode', 'edge');

%% downsample
% downsample
for n = 1:orig_data.dimInfo.nDims
    new_samplingPoints{n} = orig_data.dimInfo.samplingPoints{n}(1:2:end);
end
new_dimInfo = MrDimInfo('samplingPoints', new_samplingPoints);
down_data = orig_data.resize(new_dimInfo);
down_seg = orig_seg.resize(new_dimInfo).binarize(0.2);
% plot and check
down_data.plot('z', 13, 'displayRange', [1 255]);
down_data.plot('z', 13, 'overlayImages', down_seg, 'displayRange', [1 255]);

%% add lower intensity outline
down_seg_outline = down_seg - down_seg.imerode(strel('sphere', 1), '3d');
down_data_outline = down_data - down_seg_outline.*down_data.*0.2;

down_data_outline.plot('z', 13, 'displayRange', [1 255]);


%% add more noise
noise_image = MrImage(randn(down_data_outline.dimInfo.nSamples).*50);
noise_data = down_data_outline + noise_image.threshold(0);
noise_data.plot('z', 13);
noise_data.plot('z', 13, 'overlayImages', down_seg, 'overlayMode', 'edge');

noise_data.parameters.save.path = '/scratch/cvl/uqsboll2/temp';
