import torch
import torch.nn as nn
import numpy as np

"""
fix the anchor with trained regression
"""
class generate_predict_boxes(nn.Module):

    def __init__(self, mean=None, std=None):
        super(generate_predict_boxes, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = std

    def forward(self, anchors, regressions):

        widths  = anchors[:, :, 2] - anchors[:, :, 0]
        heights = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x   = anchors[:, :, 0] + 0.5 * widths
        ctr_y   = anchors[:, :, 1] + 0.5 * heights

        dx = regressions[:, :, 0] * self.std[0] + self.mean[0]
        dy = regressions[:, :, 1] * self.std[1] + self.mean[1]
        dw = regressions[:, :, 2] * self.std[2] + self.mean[2]
        dh = regressions[:, :, 3] * self.std[3] + self.mean[3]

        predict_ctr_x = ctr_x + dx * widths
        predict_ctr_y = ctr_y + dy * heights
        predict_w     = torch.exp(dw) * widths
        predict_h     = torch.exp(dh) * heights

        predict_boxes_x1 = predict_ctr_x - 0.5 * predict_w
        predict_boxes_y1 = predict_ctr_y - 0.5 * predict_h
        predict_boxes_x2 = predict_ctr_x + 0.5 * predict_w
        predict_boxes_y2 = predict_ctr_y + 0.5 * predict_h

        predict_boxes = torch.stack([predict_boxes_x1, predict_boxes_y1, predict_boxes_x2, predict_boxes_y2], dim=2)

        return predict_boxes

"""
adjust the predict box to the image
"""
class adjust_boxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(adjust_boxes, self).__init__()

    def forward(self, boxes, image):
        batch_size, num_channels, height, width = image.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def nms(boxes, threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # calculate area of each box
    areas = (x2 - x1) * (y2 - y1)
    # setup order
    order = scores.argsort()
    print(order)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # find intersection boxes
        intersect_x1 = np.maximum(x1[i], x1[order[1:]])
        intersect_y1 = np.maximum(y1[i], y1[order[1:]])
        intersect_x2 = np.minimum(x2[i], x2[order[1:]])
        intersect_y2 = np.minimum(y2[i], y2[order[1:]])

        # calculate the area of intersection boxes
        intersect_w = np.maximum(0.0, intersect_x2 - intersect_x1)
        intersect_h = np.maximum(0.0, intersect_y2 - intersect_y1)
        inter = intersect_h * intersect_w
        # calculate IoU
        IoU = inter / (areas[i] + areas[order[1:]] - inter)

        # find boxes needed to suppress
        inds = np.where(IoU <= threshold)[0]
        # update order
        order = order[inds + 1]
    return keep
