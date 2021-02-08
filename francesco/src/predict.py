import tensorflow as tf
import nibabel as nib
import argparse
import numpy as np

from tf_utils import misc
from model import seg_solver, unet
from preprocessing import data_loader


def main(ckp_path, ckp_name, input_path, label_path, threshold):
    assert (ckp_path[-1] == "/")
    base_path = misc.get_base_path(training=False)
    params = misc.load_json(base_path + ckp_path + "params.json")

    network = unet.UNet(params["out_ch"], params["n_layers"], params["starting_filters"], params["k_size"], params["kernel_initializer"], params["batch_norm"],
                        params["dropout"], tf.keras.layers.LeakyReLU, params["conv_per_layer"], params["max_pool"], params["upsampling"],
                        params["kernel_regularizer"])
    network.load_weights(base_path + ckp_path + ckp_name)

    # Load prediction data
    x, y, affine, header, pad = data_loader.load_testing_volume(base_path, input_path, label_path)

    slv = seg_solver.Solver(None, params, ["predicting"])
    if y is not None:
        x = x / float(threshold)
        pred = np.zeros_like(x)

        for j in range(0, x.shape[1], 128):
            for k in range(0, x.shape[2], 128):
                pred[:, j:j + 128, k:k + 128, :, :], metrics = slv.test_step(network, x[:, j:j + 128, k:k + 128, :, :], tf.keras.utils.to_categorical(tf.cast(y[:, j:j + 128, k:k + 128, :, :], tf.int32), params["out_ch"]))
                print(metrics)
                print(np.unique(y[:, j:j + 128, k:k + 128, :, :]))
    else:
        x = x / float(threshold)
        pred = np.zeros_like(x)

        for j in range(0, x.shape[1], 128):
            for k in range(0, x.shape[2], 128):
                pred[:, j:j + 128, k:k + 128, :, :] = slv.test_step(network, x[:, j:j + 128, k:k + 128, :, :], y)

    pred = pred[0, pad[0][0]:-pad[0][1], pad[1][0]:-pad[1][1], pad[2][0]:-pad[2][1], 0]
    file_name = base_path + ckp_path + str(input_path.split("/")[-1].split(".")[0]) + "-" + ckp_name + "-" + str(threshold) + ".nii.gz"
    nib.save(nib.Nifti1Image(pred, affine, header), file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckp_path", type=str, required=True)
    parser.add_argument("--ckp_name", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=False)
    parser.add_argument("--threshold", type=int, required=True)
    args = parser.parse_args()

    main(args.ckp_path, args.ckp_name, args.input_path, args.label_path, args.threshold)
