from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler


import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image


class CSV_Dataset(Dataset):
    def __init__(self, annotation_file, classes_file, transform=None):
        self.annotations_file = annotation_file
        self.classes_file = classes_file

        with open(self.classes_file, 'r') as file:
            self.classes = self.load_classes(csv.reader(file, delimiter=','))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        with open(self.annotations_file, 'r') as file:
            self.image_data = self.load_annotations(csv.reader(file, delimiter=','))

        self.image_names = list(self.image_data)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = self.read_image(index)
        annotations = self.read_annotations(index)
        sample = {'image': image, 'annotations': annotations}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_classes(self, csv_reader):
        result = {}

        for data in csv_reader:

            class_name, class_id = data
            class_id = int(class_id)

            result[class_name] = class_id

        return result

    def load_annotations(self, csv_reader):
        result = {}

        for data in csv_reader:
            image_file, x1, y1, x2, y2, class_name = data

            if image_file not in result:
                result[image_file] = []

            if(x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            result[image_file].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class_name': class_name})

        return result

    def read_image(self, image_name):
        image = skimage.io.imread(self.image_names[image_name])

        if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)

        return image.astype(np.float32) / 255.0

    def read_annotations(self, image_name):
        annotation_list = self.image_data[self.image_names[image_name]]
        annotations = np.zeros((0, 5))

        if len(annotation_list) == 0:
            return annotations

        for a in annotation_list:
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            annotation = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.classes[a['class_name']]
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)

    def label_to_name(self, label):
        return self.labels[label]


class Resizer(object):
    """
    reshape the image to an acceptabel size
    """

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annotations = sample['image'], sample['annotations']

        rows, cols, cnls = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annotations[:, :4] *= scale

        return {'image': torch.from_numpy(new_image), 'annotations': torch.from_numpy(annotations), 'scale': scale}


def collater(data):
    imgs = [s['image'] for s in data]
    annots = [s['annotations'] for s in data]
    scales = [s['scale'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'image': padded_imgs, 'annotations': annot_padded, 'scale': scales}


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotations']

        return {'image': ((image.astype(np.float32) - self.mean) / self.std), 'annotations': annotations}


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
