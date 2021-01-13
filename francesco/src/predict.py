import tensorflow as tf
import nibabel as nib
import argparse
import numpy as np

from tf_utils import misc
from model import seg_solver, unet
from preprocessing import data_loader


def main(ckp_path, ckp_name, input_path, label_path):
    assert (ckp_path[-1] == "/")
    base_path = misc.get_base_path(training=False)
    params = misc.load_json(base_path + ckp_path + "params.json")

    network = unet.UNet(params["out_ch"], params["n_layers"], params["starting_filters"], params["k_size"], params["kernel_initializer"], params["batch_norm"],
                        params["dropout"], tf.keras.layers.LeakyReLU, params["conv_per_layer"], params["max_pool"], params["upsampling"],
                        params["kernel_regularizer"])
    network.load_weights(base_path + ckp_path + ckp_name)

    # Load prediction data
    x, y, affine, header = data_loader.load_testing_volume(base_path, input_path, label_path)

    slv = seg_solver.Solver(None, params, ["predicting"])

    # Save NII file
    y = tf.keras.utils.to_categorical(tf.cast(y, tf.int32), params["out_ch"])
    for threshold in range(400, 1000, 100):
        predictions, metrics = slv.test_step(network, np.copy(x) / float(threshold), y)
        print(threshold, metrics)
        nib.save(nib.Nifti1Image(predictions[0, 28:-27, 3:-2, 38:-38, 0], affine, header), base_path + ckp_path + str(input_path.split("/")[-1][:-7]) + "-" +
                 ckp_name + "-" + str(threshold) + ".nii.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckp_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=True)
    args = parser.parse_args()

    for ckp_name in ["epoch-10"]:
        print("\n\n", ckp_name)
        main(args.ckp_path, ckp_name, args.input_path, args.label_path)
