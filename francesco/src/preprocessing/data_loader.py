import glob
import nibabel as nib
import numpy as np
import os
from intensity_normalization.normalize import nyul
from intensity_normalization.utilities import io

from tf_utils import misc, TFRecordsManager


def create_TF_records(data_path):
    base_path = misc.get_base_path(training=False)

    TFManager = TFRecordsManager()
    data_purposes = ["train", "val", "test"]
    params = {"data_purposes": data_purposes, "data_keys": {"x": "float32", "y": "float32"}}
    TFRecords_path = misc.create_TF_records_folder(base_path + data_path, data_purposes)
    misc.save_json(TFRecords_path + "params.json", params)

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
                assert (x.max() <= 255. and x.min() >= 0. and x.shape == y.shape and y.max() == 1 and y.min() == 0. and x.shape == (325, 304, 600))
                x = x / 255.

                x_patches = misc.make_patches(x, [(29, 30), (40, 40), (20, 20)], 128)
                y_patches = misc.make_patches(y, [(29, 30), (40, 40), (20, 20)], 128)
                assert (len(x_patches) == len(y_patches))

                for k in range(len(x_patches)):
                    if np.count_nonzero(y_patches[k]) == 0.:
                        continue
                    data.append({"x": np.float32(np.expand_dims(x_patches[k], -1)), "y": np.float32(y_patches[k])})

            elif "test" in data_purpose:
                x_normalized = nyul.do_hist_norm(io.open_nii(x_files[i]), percs, standard_scale, mask=None)
                io.save_nii(x_normalized, TFRecords_path + "normalized.nii", is_nii=True)
                x_normalized = np.pad(x_normalized.get_fdata() / float(255.), [(95, 95), (1, 2), (16, 16)], 'constant')
                y = np.pad(nib.load(y_files[i]).get_fdata(), [(95, 95), (1, 2), (16, 16)], 'constant')
                data.append({"x": np.float32(np.expand_dims(x_normalized, -1)), "y": np.float32(y)})
            else:
                raise NotImplementedError(data_purpose)

        TFManager.save_record(TFRecords_path + data_purpose + "/0", data)
        del data


def load_testing_volume(base_path, input_path, label_path):
    input_volume = nib.load(base_path + input_path)
    x = input_volume.get_fdata()
    y = None

    if "zipCor" in input_path or "mip_" in input_path:
        assert(x.shape[0] == 1090 and x.shape[1] == 1277 and x.shape[2] == 52)
        pad = [(31, 31), (2, 1), (38, 38)]
        x = np.pad(x, pad, 'constant')
    elif "biasCor.nii" in input_path:
        assert (x.shape[0] == 1090 and x.shape[1] == 1280 and x.shape[2] == 52)
        pad = [(31, 31), (0, 0), (38, 38)]
        x = np.pad(x, pad, 'constant')
    elif "imageData" in input_path:
        pad = [(28, 27), (3, 2), (38, 38)]
        x = np.pad(x, pad, 'constant')

    x = np.float32(np.array([np.expand_dims(x, -1)]))
    if label_path:
        y = np.pad(nib.load(base_path + label_path).get_fdata(), pad, 'constant')
        y = np.float32(np.array([np.expand_dims(y, -1)]))
    assert (len(x.shape) == 5 and x.shape[0] == 1 and x.shape[-1] == 1)

    return x, y, input_volume.affine, input_volume.header, pad
