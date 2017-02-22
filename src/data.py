import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path

import pandas as pd
import numpy as np
import cv2
import imageio as io
import torch
import cPickle as pickle
import random

torch.manual_seed(2017)
np.random.seed(2017)
random.seed(2017)

size = 1024
dim = 256
off = (size - dim)//2


def transform(img, seg):

    angle = np.random.uniform(-5, 5)
    scale = np.random.uniform(1/1.1, 1.1)
    size = img.shape[0:2]
    img_center = tuple(np.array(size)/2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, scale)
    rot_mat[:,2] += np.random.uniform(-30, 30, (2,))

    img = cv2.warpAffine(img, rot_mat, size, flags=cv2.INTER_CUBIC)
    seg = cv2.warpAffine(seg, rot_mat, size, flags=cv2.INTER_CUBIC)

    seg = seg > 0.3

    return img, seg


class ImageList(data.Dataset):
    def __init__(self, df, root='', for_train=False):
        self.root = root
        self.df = df
        self.for_train = for_train

    def __getitem__(self, index):
        fname, right, x, y, a, b, t = self.df.iloc[index, :]
        center = (int(x), int(y))
        axes = (int(a), int(b))

        image = cv2.imread(self.root + fname, 0)

        mask = np.zeros(image.shape, dtype='float32')
        cv2.ellipse(mask, center, axes, t*180/np.pi, 0, 360, 1.0, 1)

        if right:
            image = image[:,::-1]
            mask = mask[:, ::-1]


        image = image[None, :, :].astype('float32')/255.0
        mask = mask[None, :, :].astype('float32')

        return torch.from_numpy(image), torch.from_numpy(mask)

    def __len__(self):
        return self.df.shape[0]
