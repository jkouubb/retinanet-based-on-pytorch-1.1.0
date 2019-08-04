import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self):
        super(Anchors, self).__init__()

        self.pyramid_levels = [3, 4, 5, 6, 7]
        #calcu the scale that feature map in different level needs in order to map to the original image
        self.strides = [2 ** x for x in self.pyramid_levels]
        #calcu the basic anchor size in each level
        self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]

    def forward(self, image):

        #get image's width and height
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        #calcu feature map's size
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            #generate anchors in different sizes
            anchors = generate_anchors(base_anchor_size=self.sizes[idx])
            #map the anchors to the original image
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            #put anchors in different level together
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        #output:matrix in size of (map_width * map_height * anchor_num) * 4
        return torch.from_numpy(all_anchors.astype(np.float32)).cuda()


def generate_anchors(base_anchor_size=16):
    #with three ratios and three scales, the function can generate 9 different anchors


    ratios = np.array([0.5, 1, 2])
    scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_anchor_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr=0, y_ctr=0, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    #map to the oringinal image
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    #get a matrix which descirbes a possible square(which is decribed with(x1, y1, x2, y2))
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()


    #all 9 anchors can cover each square, so there is map_width * map_height *anchor_num possible anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

