import glob
import nibabel as nib
import numpy as np
from intensity_normalization.normalize import nyul
from intensity_normalization.utilities import io

from tf_utils import misc, TFRecordsManager


def create_TF_records(data_path, normalize):
    base_path = misc.get_base_path(training=False)

    TFManager = TFRecordsManager()
    data_purposes = ["train", "val", "test"]
    params = {"data_purposes": data_purposes, "data_keys": {"x": "float32", "y": "float32"}, "normalize": normalize, "data_path": data_path}
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
                if normalize:
                    assert (x.max() <= 255. and x.min() >= 0. and x.shape == y.shape and y.max() == 1 and y.min() == 0. and "original" in data_path)
                    x /= float(255.)

                if "original" in data_path:
                    assert(x.shape == (325, 304, 600))
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
                x_normalized = x_normalized.get_fdata()
                if normalize:
                    assert ("original" in data_path)
                    x_normalized /= float(255.)
                assert(x_normalized.shape == (1090, 1277, 32))
                x_normalized = np.pad(x_normalized, [(95, 95), (1, 2), (16, 16)], 'constant')
                y = np.pad(nib.load(y_files[i]).get_fdata(), [(95, 95), (1, 2), (16, 16)], 'constant')
                data.append({"x": np.float32(np.expand_dims(x_normalized, -1)), "y": np.float32(y)})
            else:
                raise NotImplementedError(data_purpose)

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
