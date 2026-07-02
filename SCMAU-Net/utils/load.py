import os
import re
import torch
import scipy.io as sio
import numpy as np
from PIL import Image
import scipy.io as io
from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""

    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""

    for id, pos in ids:

        im = io.loadmat(dir + id + suffix)['imags']
        yield im
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def get_name(dir):
    return (f[:-4] for f in os.listdir(dir))

def str2int(v_str):


    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def cropped_imgs(ids, dir_img):
    dirsimgs = sort_humanly(os.listdir(dir_img))

    name = sort_humanly(get_name(dir_img))
    for i in range(len(list(ids))):
        imgs = sio.loadmat(dir_img+dirsimgs[i])[name[i]]

        yield imgs

def sort_humanly(v_list):
    return sorted(v_list, key=str2int)

def cropped_masks(ids, dir_mask):
    dirsmasks = sort_humanly(os.listdir(dir_mask))
    name = sort_humanly(get_name(dir_mask))
    for i in range(len(list(ids))):
        masks = sio.loadmat(dir_mask+dirsmasks[i])[name[i]]
        yield masks

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.mat', scale)

    masks = to_cropped_imgs(ids, dir_mask, '.mat', scale)

    return zip(imgs, masks)

def load_mat_imgs_masks(ids, dir_img, dir_mask):
    imgs = cropped_imgs(ids, dir_img)

    masks = cropped_masks(ids, dir_mask)
    return zip(imgs, masks)

