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

OUT_DIR = 'tmp/experiment2_aspp_aug'
XT = 'tmp/imageData.nii'
YT = 'tmp/segmentationData.nii'
N_STEPS = 10000
PATCH_SIZE = 50

DATA2 = '/afm02/Q3/Q3503/synthetic/aug'
DATA = '/afm02/Q3/Q3503/synthetic/raw'
SEG = '/afm02/Q3/Q3503/synthetic/seg'






if __name__ == "__main__":
    # set up dirs... WARNING: will delete everything 
    force_create_empty_directory(OUT_DIR)
    force_create_empty_directory(os.path.join(OUT_DIR, 'pred'))

    xt = nib.load(XT).get_fdata()
    yt = nib.load(YT).get_fdata()

    g1 = data_gen(DATA, SEG, patch_size=PATCH_SIZE)
    g2 = data_gen(DATA2, SEG, patch_size=PATCH_SIZE)
    gs = [g1, g2]
    m = Model(5, PATCH_SIZE).cuda()
    # OPTIONAL, load weights
    # m.load_state_dict(torch.load('tmp/model.pth'))

    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    losses = []

    # pipe stdout to a file
    sys.stdout = open(os.path.join(OUT_DIR, 'out.log'), 'w+')

    # pip stderr to a file
    sys.stderr = open(os.path.join(OUT_DIR, 'err.log'), 'w+')
    for i in range(N_STEPS): # train for 10k steps
        g = random.choice(gs) # choose randomly from aug and raw
        # training step: single iteration of backprop
        xp, yp = next(g)
        opt.zero_grad()
        o = m(xp)
        loss = F.binary_cross_entropy_with_logits(o, yp)
        loss.backward()
        opt.step()
        losses.append(loss.cpu().detach().numpy())
        print(i, np.mean(losses[-50:]))
        sys.stdout.flush() # remember to flush the prints, else nothing is going to appear until job ends

        # eval and save every 50 steps
        if i % 50 == 0: 
            torch.save(m.state_dict(), os.path.join(OUT_DIR, 'model.pth')) # save model weights
            m.eval()
            for i in range(5):
                xp_save, yp = get_patch(xt, yt)
                xp = torch.tensor(xp_save.astype('float32')).unsqueeze(0).cuda()
                pred = m(xp)
                pred = torch.sigmoid(pred)
                arr = pred[0][0].cpu().detach().numpy()
                save_as_nifti(xp_save[0], os.path.join(OUT_DIR, 'pred'), '%d_x.nii.gz'% i)
                save_as_nifti(arr, os.path.join(OUT_DIR, 'pred'), '%d_pred.nii.gz'% i)
                save_as_nifti(np.round(arr), os.path.join(OUT_DIR, 'pred'), '%d_pred_rounded.nii.gz'% i)
                save_as_nifti(yp[0], os.path.join(OUT_DIR, 'pred'), '%d_y.nii.gz'% i)
            m.train()
        


