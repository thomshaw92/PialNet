import torch
import torch.nn as nn
import torch.nn.functional as F

import nibabel as nib
import numpy as np
import os
import random
import scanf
import sys
from models import *
from utils import *
from losses import compute_per_channel_dice

OUT_DIR = 'tmp/experiment_hist'
XT = '/afm02/Q3/Q3461/data/validation_data_sensible_naming/1.nii.gz'
YT = '/afm02/Q3/Q3461/data/MRA_P11/vessel_segmentation_intensity/seg_TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_21_biasCor_zipCor_H375_L250_C5.nii'
N_STEPS = 10000
PATCH_SIZE = 50

DATA2 = '/afm02/Q3/Q3503/synthetic/aug'
DATA = '/afm02/Q3/Q3503/synthetic/raw'
SEG = '/afm02/Q3/Q3503/synthetic/seg'

USE_DSC = True


if __name__ == "__main__":

    # load data
    xt = nib.load(XT).get_fdata()
    yt = nib.load(YT).get_fdata()
    print(xt.shape)
    print(yt.shape)
    m = Model(5, PATCH_SIZE)
    #load weights
    m.load_state_dict(torch.load('/afm02/Q3/Q3503/synthetic/checkpoints/hist.pth', map_location={'cuda:0': 'cpu'}))  

    # apply
    m.eval()
    for i in range(50):
        xp_save, yp = get_patch(xt, yt, norm=True)
        xp = torch.tensor(xp_save.astype('float32')).unsqueeze(0)
        pred = m(xp)
        pred = torch.sigmoid(pred)
        arr = pred[0][0].cpu().detach().numpy()
        # print(np.unique(arr))
        print(np.min(arr), np.max(arr))
        save_as_nifti(xp_save[0], os.path.join(OUT_DIR, 'pred'), '%d_x.nii.gz'% i)
        save_as_nifti(arr, os.path.join(OUT_DIR, 'pred'), '%d_pred.nii.gz'% i)
        save_as_nifti(np.round(arr), os.path.join(OUT_DIR, 'pred'), '%d_pred_rounded.nii.gz'% i)
        save_as_nifti(yp[0], os.path.join(OUT_DIR, 'pred'), '%d_y.nii.gz'% i)
    

