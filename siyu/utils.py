import torch
import torch.nn as nn
import torch.nn.functional as F

import nibabel as nib
import numpy as np
import os
import random
import scanf
import sys
import shutil

def force_create_empty_directory(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

def get_patch(img, seg, size=50):
    def rand_seg(d, l):
        lower = 0
        upper = d - l
        idx = random.randint(lower, upper)
        idx = random.randint(lower, upper)
        return idx, idx + l
    d1, d2, d3 = img.shape
    s1, s2, s3 = [rand_seg(d, size) for d in [d1, d2, d3]]
    l1, u1 = s1
    l2, u2 = s2
    l3, u3 = s3
    imgp, segp = img[l1:u1, l2:u2, l3:u3], seg[l1:u1, l2:u2, l3:u3]
    imgp = imgp[None, ...]
    segp = segp[None, ...]
    return (imgp - imgp.mean()) / imgp.std(), segp


DATA = '/afm02/Q3/Q3503/synthetic/raw'
SEG = '/afm02/Q3/Q3503/synthetic/seg'

def data_gen(data_dir, seg_dir, images_per_batch=1, patches_per_img=4, patch_size=50):
    data_files = os.listdir(data_dir)
    cases = {}
    for f in data_files:
        case_num = scanf.scanf('%d.nii.gz', f)[0]
        cases[case_num] = [os.path.join(data_dir, f)]

    seg_files = os.listdir(seg_dir)
    for f in seg_files:
        case_num = scanf.scanf('%d.nii.gz', f)[0]
        cases[case_num].append(os.path.join(seg_dir, f))
        assert len(cases[case_num]) == 2
    
    data = list(cases.values())
    while True:
        batch = random.choices(data, k=images_per_batch)
        xb, yb = [], []
        for x, y in batch:
            x = nib.load(x).get_fdata()
            y = nib.load(y).get_fdata()
            for i in range(patches_per_img):         
                xp, yp = get_patch(x, y, size=patch_size)
                xb.append(xp)
                yb.append(yp)
        xb = np.array(xb).astype('float32')
        yb = np.array(yb)
        yield torch.tensor(xb).cuda(), torch.tensor(yb).cuda()

def save_as_nifti(data, folder, name, affine=np.eye(4)):
    img = nib.Nifti1Image(data, affine)
    if not os.path.exists(folder):
        os.mkdir(folder)
    nib.save(img, os.path.join(folder, name))