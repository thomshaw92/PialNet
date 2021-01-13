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

OUT_DIR = 'tmp/experiment2_aspp_dsc'
XT = 'tmp/imageData.nii'
YT = 'tmp/segmentationData.nii'
N_STEPS = 10000
PATCH_SIZE = 50

DATA2 = '/afm02/Q3/Q3503/synthetic/aug'
DATA = '/afm02/Q3/Q3503/synthetic/raw'
SEG = '/afm02/Q3/Q3503/synthetic/seg'

USE_DSC = True


if __name__ == "__main__":

    m = Model(5, PATCH_SIZE)
    # OPTIONAL, load weights
    m.load_state_dict(torch.load('tmp/model.pth'))  

        # apply
            m.eval()
            for i in range(5):
                xp_save, yp = get_patch(xt, yt)
                xp = torch.tensor(xp_save.astype('float32')).unsqueeze(0)
                pred = m(xp)
                pred = torch.sigmoid(pred)
                arr = pred[0][0].cpu().detach().numpy()
                save_as_nifti(xp_save[0], os.path.join(OUT_DIR, 'pred'), '%d_x.nii.gz'% i)
                save_as_nifti(arr, os.path.join(OUT_DIR, 'pred'), '%d_pred.nii.gz'% i)
                save_as_nifti(np.round(arr), os.path.join(OUT_DIR, 'pred'), '%d_pred_rounded.nii.gz'% i)
                save_as_nifti(yp[0], os.path.join(OUT_DIR, 'pred'), '%d_y.nii.gz'% i)
            m.train()
        


