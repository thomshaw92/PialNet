import glob
import nibabel as nib
import numpy as np

from tf_utils import misc, TFRecordsManager


def create_TF_records(data_path):
    base_path = misc.get_base_path(training=False)

    TFManager = TFRecordsManager()
    data_purposes = ["train", "val"]
    data_purposes += ["test-" + str(threshold) for threshold in range(400, 1000, 100)]
    params = {"data_purposes": data_purposes, "data_keys": {"x": "float32", "y": "float32"}}

    TFRecords_path = misc.create_TF_records_folder(base_path + data_path, data_purposes)
    misc.save_json(TFRecords_path + "params.json", params)

    for data_purpose in data_purposes:
        if "test" not in data_purpose:
            x_files = glob.glob(base_path + data_path + data_purpose + "/raw/*")
            y_files = glob.glob(base_path + data_path + data_purpose + "/seg/*")
        else:
            x_files = glob.glob(base_path + data_path + "test/raw/*")
            y_files = glob.glob(base_path + data_path + "test/seg/*")
        x_files.sort()
        y_files.sort()
        assert (len(x_files) == len(y_files))
        data = []
        for i in range(len(x_files)):
            x = nib.load(x_files[i]).get_fdata()
            y = nib.load(y_files[i]).get_fdata()

            if "train" == data_purpose or "val" == data_purpose:
                assert (x.max() == 255. and x.min() == 0. and y.max() == 1.0 and y.min() == 0.)
                x = x / 255.
                assert (x.max() == 1. and x.min() == 0.)

                x_patches = misc.make_patches(x, [(29, 30), (40, 40), (20, 20)], 128)
                y_patches = misc.make_patches(y, [(29, 30), (40, 40), (20, 20)], 128)
                assert (len(x_patches) == len(y_patches))

                for k in range(len(x_patches)):
                    data.append({"x": np.float32(np.expand_dims(x_patches[k], -1)), "y": np.float32(y_patches[k])})

            elif "test" in data_purpose:
                x = np.pad(x, [(28, 27), (3, 2), (38, 38)], 'constant')
                y = np.pad(y, [(28, 27), (3, 2), (38, 38)], 'constant')
                data.append({"x": np.float32(np.expand_dims((np.copy(x) / float(data_purpose.split("-")[1])), -1)), "y": np.float32(y)})
            else:
                raise NotImplementedError(data_purpose)

        TFManager.save_record(TFRecords_path + data_purpose + "/0", data)
        del data


def create_TF_records_augmented_data(data_path):
    base_path = misc.get_base_path(training=False)

    TFManager = TFRecordsManager()
    data_purposes = ["train_aug", "val_aug"]
    data_purposes += ["test-" + str(threshold) for threshold in range(400, 1000, 100)]
    params = {"data_purposes": data_purposes, "data_keys": {"x": "float32", "y": "float32"}}

    TFRecords_path = misc.create_TF_records_folder(base_path + data_path, data_purposes)
    misc.save_json(TFRecords_path + "params.json", params)

    for data_purpose in data_purposes:
        if "test" not in data_purpose:
            x_files = glob.glob(base_path + data_path + data_purpose + "/aug/*")
            y_files = glob.glob(base_path + data_path + data_purpose + "/aug_seg/*")
        else:
            x_files = glob.glob(base_path + data_path + "test/raw/*")
            y_files = glob.glob(base_path + data_path + "test/seg/*")
        x_files.sort()
        y_files.sort()
        assert (len(x_files) == len(y_files))
        data = []
        for i in range(len(x_files)):
            x = nib.load(x_files[i]).get_fdata()
            y = nib.load(y_files[i]).get_fdata()

            if "train" in data_purpose or "val" in data_purpose:
                assert (x.shape[0] == 163 and x.shape[1] == 152 and x.shape[2] == 300)
                assert (y.shape[0] == 163 and y.shape[1] == 152 and y.shape[2] == 300)
                x = x / 255.

                x_patches = misc.make_patches(x, [(47, 46), (52, 52), (42, 42)], 128)
                y_patches = misc.make_patches(y, [(47, 46), (52, 52), (42, 42)], 128)
                assert (len(x_patches) == len(y_patches))

                for k in range(len(x_patches)):
                    data.append({"x": np.float32(np.expand_dims(x_patches[k], -1)), "y": np.float32(y_patches[k])})

            elif "test" in data_purpose:
                x = np.pad(x, [(28, 27), (3, 2), (38, 38)], 'constant')
                y = np.pad(y, [(28, 27), (3, 2), (38, 38)], 'constant')
                data.append({"x": np.float32(np.expand_dims((np.copy(x) / float(data_purpose.split("-")[1])), -1)), "y": np.float32(y)})
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
