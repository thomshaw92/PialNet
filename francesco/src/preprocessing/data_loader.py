import glob
import nibabel as nib
import numpy as np
from intensity_normalization.normalize import nyul
from intensity_normalization.utilities import io

from tf_utils import misc, TFRecordsManager


def create_records_original_dataset(data_path="dataset/original/", data_purposes=["train", "val", "test"]):
    # TFRecords set up with paths
    base_path = misc.get_base_path(training=False)
    TFManager = TFRecordsManager()
    TFRecords_path = misc.create_TF_records_folder(base_path + data_path, data_purposes)
    misc.save_json(TFRecords_path + "params.json", {"data_purposes": data_purposes, "data_keys": {"x": "float32", "y": "float32"},
                                                    "data_path": data_path, "type": "create_records_original_dataset"})
    # Train normalizer to match training data
    ref_files = glob.glob(base_path + data_path + "train/raw/*")
    mask_files = [None] * len(ref_files)
    standard_scale, percs = nyul.train(ref_files, mask_files)
    misc.save_pickle(TFRecords_path + "norm.pkl", {"standard_scale": standard_scale, "percs": percs})

    for data_purpose in data_purposes:
        x_files = glob.glob(base_path + data_path + data_purpose + "/raw/*")
        y_files = glob.glob(base_path + data_path + data_purpose + "/seg/*")
        x_files.sort()
        y_files.sort()
        assert (len(x_files) == len(y_files))
        data = []
        for i in range(len(x_files)):

            if "train" in data_purpose or "val" in data_purpose:
                x = nib.load(x_files[i]).get_fdata()
                y = nib.load(y_files[i]).get_fdata()
                assert (x.max() <= 255. and x.min() >= 0. and x.shape == y.shape and y.max() == 1 and y.min() == 0.)
                x /= float(255.)

                # Make patches
                assert (x.shape == (325, 304, 600))
                x_patches = misc.make_patches(x, [(29, 30), (40, 40), (20, 20)], 128)
                y_patches = misc.make_patches(y, [(29, 30), (40, 40), (20, 20)], 128)
                for k in range(len(x_patches)):
                    if np.count_nonzero(y_patches[k]) > 100:
                        data.append({"x": np.float32(np.expand_dims(x_patches[k], -1)), "y": np.float32(y_patches[k])})
                del x_patches
                del y_patches
            elif "test" in data_purpose:
                x_normalized = nyul.do_hist_norm(io.open_nii(x_files[i]), percs, standard_scale, mask=None)
                # io.save_nii(x_normalized, TFRecords_path + "normalized.nii", is_nii=True)
                x_normalized = x_normalized.get_fdata() / float(255.)
                assert (x_normalized.shape == (1090, 1277, 32))
                x_normalized = np.pad(x_normalized[:, :, 16:], [(95, 95), (1, 2), (0, 0)], 'constant')
                y = np.pad(nib.load(y_files[i]).get_fdata()[:, :, 16:], [(95, 95), (1, 2), (0, 0)], 'constant')
                data.append({"x": np.float32(np.expand_dims(x_normalized, -1)), "y": np.float32(y)})
            else:
                raise NotImplementedError(data_purpose)

        TFManager.save_record(TFRecords_path + data_purpose + "/0", data)
        del data


def create_records_original_dataset_half_manual(data_path="dataset/original/", data_purposes=["train", "val", "test"]):
    # TFRecords set up with paths
    base_path = misc.get_base_path(training=False)
    TFManager = TFRecordsManager()
    TFRecords_path = misc.create_TF_records_folder(base_path + data_path, data_purposes)
    misc.save_json(TFRecords_path + "params.json", {"data_purposes": data_purposes, "data_keys": {"x": "float32", "y": "float32"},
                                                    "data_path": data_path, "type": "create_records_original_dataset_half_manual"})
    # Train normalizer to match training data
    ref_files = glob.glob(base_path + data_path + "train/raw/*")
    mask_files = [None] * len(ref_files)
    standard_scale, percs = nyul.train(ref_files, mask_files)
    misc.save_pickle(TFRecords_path + "norm.pkl", {"standard_scale": standard_scale, "percs": percs})

    # Normalize test volume and load manual segmentation
    name = "TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_denoised_2SR_resampled_22-52_masked.nii"
    x_normalized = nyul.do_hist_norm(io.open_nii(base_path + data_path + "test/raw/" + name), percs, standard_scale, mask=None).get_fdata()
    x_normalized /= float(255.)
    name = "seg_TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_27_biasCor_zipCor_H400_L300_C10_resized_22-52_masked.nii"
    y_normalized = nib.load(base_path + data_path + "test/seg/" + name).get_fdata()
    assert (x_normalized.shape == (1090, 1277, 32) and y_normalized.shape == (1090, 1277, 32))

    # Pad
    x_normalized = np.pad(x_normalized, [(95, 95), (1, 2), (0, 0)], 'constant')
    y_normalized = np.pad(y_normalized, [(95, 95), (1, 2), (0, 0)], 'constant')

    for data_purpose in data_purposes:
        x_files = glob.glob(base_path + data_path + data_purpose + "/raw/*")
        y_files = glob.glob(base_path + data_path + data_purpose + "/seg/*")
        x_files.sort()
        y_files.sort()
        assert (len(x_files) == len(y_files))
        data = []
        for i in range(len(x_files)):

            if "train" in data_purpose or "val" in data_purpose:
                x = nib.load(x_files[i]).get_fdata()
                y = nib.load(y_files[i]).get_fdata()
                assert (x.max() <= 255. and x.min() >= 0. and x.shape == y.shape and y.max() == 1 and y.min() == 0.)
                x /= float(255.)

                # Make patches
                assert (x.shape == (325, 304, 600))
                x_patches = misc.make_patches(x, [(29, 30), (40, 40), (20, 20)], 128)
                y_patches = misc.make_patches(y, [(29, 30), (40, 40), (20, 20)], 128)
                for k in range(len(x_patches)):
                    if np.count_nonzero(y_patches[k]) > 100:
                        data.append({"x": np.float32(np.expand_dims(x_patches[k], -1)), "y": np.float32(y_patches[k])})
                del x_patches
                del y_patches
            elif "test" in data_purpose:
                data.append({"x": np.float32(np.expand_dims(x_normalized[:, :, 16:], -1)), "y": np.float32(y_normalized[:, :, 16:])})
            else:
                raise NotImplementedError(data_purpose)

        if "train" in data_purpose:
            x_patches = misc.make_patches(x_normalized[:, :, :16], [(0, 0), (0, 0), (56, 56)], 128)
            y_patches = misc.make_patches(y_normalized[:, :, :16], [(0, 0), (0, 0), (56, 56)], 128)
            for k in range(len(x_patches)):
                data.append({"x": np.float32(np.expand_dims(x_patches[k], -1)), "y": np.float32(y_patches[k])})
            del x_patches
            del y_patches

        TFManager.save_record(TFRecords_path + data_purpose + "/0", data)
        del data


def load_testing_volume(paths, norm_stats_path):
    data, meta = {}, {}
    norm_pkl = misc.load_pickle(norm_stats_path)

    for label in paths:
        if paths[label] is None:
            data[label] = None
            continue

        volume = nib.load(paths[label])
        if "x" == label:
            data[label] = nyul.do_hist_norm(io.open_nii(paths[label]), norm_pkl["percs"], norm_pkl["standard_scale"], mask=None).get_fdata()
            data[label] /= float(255.)
        else:
            data[label] = volume.get_fdata()

        # Add metadata
        if "affine" not in meta:
            meta["affine"] = volume.affine
            meta["header"] = volume.header
            meta["orig_shape"] = volume.shape

            # Check if padding needed
            pad_size, is_same_shape = misc.get_how_much_to_pad(meta["orig_shape"], 256)

        # Convert to float32 and expand_dims
        data[label] = np.float32(np.array([np.expand_dims(data[label], -1)]))
        assert (len(data[label].shape) == 5 and data[label].shape[0] == 1 and data[label].shape[-1] == 1)

        # Add padding if needed
        if not is_same_shape:
            data[label], _, meta["pad_added"] = misc.add_padding(data[label], pad_size)

    return data, meta
