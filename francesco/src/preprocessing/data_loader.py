import glob
import nibabel as nib
import numpy as np

from tf_utils import misc, TFRecordsManager


def create_TF_records(data_path):
    base_path = misc.get_base_path(training=False)

    TFManager = TFRecordsManager()
    data_purposes = ["train", "val", "test"]
    params = {"data_purposes": data_purposes, "data_keys": {"x": "float32", "y": "float32"}}

    TFRecords_path = misc.create_TF_records_folder(base_path + data_path, data_purposes)
    misc.save_json(TFRecords_path + "params.json", params)

    for data_purpose in data_purposes:
        x_files = glob.glob(base_path + data_path + data_purpose + "/raw/*")
        y_files = glob.glob(base_path + data_path + data_purpose + "/seg/*")
        x_files.sort()
        y_files.sort()

        assert (len(x_files) == len(y_files))
        data = []
        for i in range(len(x_files)):
            x = nib.load(x_files[i]).get_fdata()
            y = nib.load(y_files[i]).get_fdata()

            if "train" == data_purpose or "val" == data_purpose:
                assert (x.max() == 255. and x.min() == 0. and y.max() == 1.0 and y.min() == 0.)
                x_patches = misc.make_patches(x, [(29, 30), (40, 40), (20, 20)], 128)
                y_patches = misc.make_patches(y, [(29, 30), (40, 40), (20, 20)], 128)
                assert (len(x_patches) == len(y_patches))

                for k in range(len(x_patches)):
                    data.append({"x": np.float32(np.expand_dims(x_patches[k], -1)), "y": np.float32(y_patches[k])})

            elif "test" in data_purpose:
                x = np.pad(x, [(28, 27), (3, 2), (38, 38)], 'constant')
                y = np.pad(y, [(28, 27), (3, 2), (38, 38)], 'constant')
                data.append({"x": np.float32(np.expand_dims(x, -1)), "y": np.float32(y)})
            else:
                raise NotImplementedError(data_purpose)

        TFManager.save_record(TFRecords_path + data_purpose + "/0", data)
        del data
