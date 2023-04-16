from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret


class ImageFolder(data.Dataset):
    def __init__(self, root, split_dir='train', custom_classes=None,
                 base_size=64, transform=None, target_transform=None):
        root = os.path.join(root, split_dir)
        classes, class_to_idx = self.find_classes(root, custom_classes)
        imgs = self.make_dataset(classes, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.num_classes = len(classes)
        self.class_to_idx = class_to_idx

        self.transform = transform
        self.target_transform = target_transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        print('num_classes', self.num_classes)

    def find_classes(self, dir, custom_classes):
        classes = []

        for d in os.listdir(dir):
            if os.path.isdir:
                if custom_classes is None or d in custom_classes:
                    classes.append(os.path.join(dir, d))
        print('Valid classes: ', len(classes), classes)

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, classes, class_to_idx):
        images = []
        for d in classes:
            for root, _, fnames in sorted(os.walk(d)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[d])
                        images.append(item)
        print('The number of images: ', len(images))
        return images

    def __getitem__(self, index):
        path, target = self.imgs[index]
        imgs_list = get_imgs(path, self.imsize,
                             transform=self.transform,
                             normalize=self.norm)

        return imgs_list

    def __len__(self):
        return len(self.imgs)
