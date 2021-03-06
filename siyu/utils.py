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
# import cv2

def force_create_empty_directory(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

def normalise(img):
    return (img - img.mean()) / img.std()
    # return cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)

def get_patch(img, seg, size=50, norm=True):
    img = normalise(img)
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
    return imgp, segp


# def get_patch(img, seg, size=50, norm=True):
#     def rand_seg(d, l):
#         lower = 0
#         upper = d - l
#         idx = random.randint(lower, upper)
#         idx = random.randint(lower, upper)
#         return idx, idx + l
#     d1, d2, d3 = img.shape
#     s1, s2, s3 = [rand_seg(d, size) for d in [d1, d2, d3]]
#     l1, u1 = s1
#     l2, u2 = s2
#     l3, u3 = s3
#     imgp, segp = img[l1:u1, l2:u2, l3:u3], seg[l1:u1, l2:u2, l3:u3]
#     imgp = imgp[None, ...]
#     segp = segp[None, ...]
#     if norm:
#         return normalise(imgp), segp
#     else:
#         return imgp, segp


def data_gen(data_dir, seg_dir, images_per_batch=1, patches_per_img=4, patch_size=50, norm=True):
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
                xp, yp = get_patch(x, y, size=patch_size, norm=norm)
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

def data_gen_hist(data_dir, seg_dir, images_per_batch=1, patches_per_img=4, patch_size=50, norm=False):
    data_files = os.listdir(data_dir)
    cases = {}
    for f in data_files:
        if 'nii.gz' not in f:
            continue
        case_num = scanf.scanf('%s_%s_%d.nii.gz', f)[2]
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
                xp, yp = get_patch(x, y, size=patch_size, norm=norm)
                xb.append(xp)
                yb.append(yp)
        xb = np.array(xb).astype('float32')
        yb = np.array(yb)
        yield torch.tensor(xb).cuda(), torch.tensor(yb).cuda()