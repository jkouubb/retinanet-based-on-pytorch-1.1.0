import numpy as np
import torch
import torch.nn as nn

"""
Calcu_IoU:
    input:anchors, annotations
    output:a matrix in size of anchor_num * annotation_num
    purpose: Calcu all possible matches' IoU
"""
def Calcu_IoU(anchors, annotations):
    annotation_area = (annotations[:, 2] - annotations[:, 0]) * (annotations[:, 3] - annotations[:, 1])

    intersectant_width = torch.min(torch.unsqueeze(anchors[:, 2], dim=1), annotations[:, 2]) - torch.max(torch.unsqueeze(anchors[:, 0], 1), annotations[:, 0])
    intersectant_height = torch.min(torch.unsqueeze(anchors[:, 3], dim=1), annotations[:, 3]) - torch.max(torch.unsqueeze(anchors[:, 1], 1), annotations[:, 1])

    intersectant_width = torch.clamp(intersectant_width, min=0)
    intersectant_height = torch.clamp(intersectant_height, min=0)

    union_area = torch.unsqueeze((anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1]), dim=1) + annotation_area - intersectant_width * intersectant_height

    union_area = torch.clamp(union_area, min=1e-8)

    intersection = intersectant_width * intersectant_height

    IoU = intersection / union_area

    return IoU


def Calcu_Loss(regressions, classifications, anchors, annotations):
    alpha = 0.25
    gamma = 2.0
    classification_loss = []
    regression_loss = []
    anchor = anchors[0, :, :].cuda()
    batch_size = classifications.shape[0]

    for i in range(batch_size):
        """
        part 1:
        calcu classification loss
        formula: Classification Loss(p) = -alpha * (1 - p) ** gamma * log(p)
        """
        classification = classifications[i, :, :]
        annotation = annotations[i, :, :].cuda()
        #throw empty annotations if they exist
        annotation = annotation[annotation[:, 4] != -1]

        if annotation.shape[0] == 0:
            regression_loss.append(torch.tensor(0).float().cuda())
            classification_loss.append(torch.tensor(0).float().cuda())
            continue

        classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
        IoU = Calcu_IoU(anchor, annotation[:, :4])
        IoU_max, annotation_index = torch.max(IoU, dim=1)

        """
        setup judge matrix
        if an anchor is positive, the matrix will puts 1 on the possible annotation position
        if an anchor is negtive, the matrix will puts 0s on the whole line
        otherwise, the matrix will puts -1s on the whole line
        """
        judge_matrix = torch.ones(classification.shape) * -1
        judge_matrix = judge_matrix.cuda()

        judge_matrix[torch.lt(IoU_max, 0.4), :] = 0

        valid_annotation = annotation[annotation_index, :]
        positive_indices = torch.ge(IoU_max, 0.5)

        judge_matrix[positive_indices, :] = 0
        judge_matrix[positive_indices, valid_annotation[positive_indices, 4].long()] = 1

        alpha_matrix = torch.ones(judge_matrix.shape).cuda() * alpha
        alpha_matrix = torch.where(torch.eq(judge_matrix, 1.), alpha_matrix, 1. - alpha_matrix)

        """
        setup possibility matrix ans calcu the '-alpah * (1 - p) ** gamma' part
        """
        possibility_matrix = torch.where(torch.eq(judge_matrix, 1.), 1. - classification, classification)
        possibility_matrix = alpha_matrix * torch.pow(possibility_matrix, gamma)

        """
        do the rest part and put the loss into the final output classification loss
        """
        cl = -(judge_matrix * torch.log(possibility_matrix) + (1.0 - judge_matrix) * torch.log(1.0 - possibility_matrix))
        cl = possibility_matrix * cl
        cl = torch.where(torch.ne(judge_matrix, -1.0), cl, torch.zeros(cl.shape).cuda())
        classification_loss.append(cl.sum() / torch.clamp(positive_indices.sum().float(), min=1.0))

        """
        part 2:
        calcu the regression loss
        formula: Regression Loss(regression) = Smooth(destiny regression - regression)
                 Smooth(x) = 0.5 * x ** 2 (abs(x) <= 1) or abs(x) - 0.5 (abs(x)>1)
        """
        if positive_indices.sum() > 0:
            positive_annotation = valid_annotation[positive_indices, :]
            anchor_widths = anchor[:, 2] - anchor[:, 0]
            anchor_heights = anchor[:, 3] - anchor[:, 1]
            anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
            anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

            positive_anchor_widths = anchor_widths[positive_indices]
            positive_anchor_heights = anchor_heights[positive_indices]
            positive_anchor_ctr_x = anchor_ctr_x[positive_indices]
            positive_anchor_ctr_y = anchor_ctr_y[positive_indices]

            ground_truth_widths = positive_annotation[:, 2] - positive_annotation[:, 0]
            ground_truth_heights = positive_annotation[:, 3] - positive_annotation[:, 1]
            ground_truth_ctr_x = positive_annotation[:, 0] + 0.5 * ground_truth_widths
            ground_truth_ctr_y = positive_annotation[:, 1] + 0.5 * ground_truth_heights

            #prevent the ground truth's size is too small
            torch.clamp(ground_truth_widths, min=1)
            torch.clamp(ground_truth_heights, min=1)

            #calcu the destiny regression
            destiny_regression_x = (ground_truth_ctr_x - positive_anchor_ctr_x) / positive_anchor_widths
            destiny_regression_y = (ground_truth_ctr_y - positive_anchor_ctr_y) / positive_anchor_heights
            destiny_regression_widths = torch.log(ground_truth_widths / positive_anchor_widths)
            destiny_regression_heights = torch.log(ground_truth_heights / positive_anchor_heights)

            destiny_regression = torch.stack((destiny_regression_x, destiny_regression_y, destiny_regression_widths, destiny_regression_heights))
            destiny_regression = destiny_regression.t()

            regression =regressions[i, :, :]
            destiny_regression = destiny_regression / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
            regression_delta = torch.abs(destiny_regression - regression[positive_indices, :])
            #since a group of 9 anchors share the same center, the parameters in function Smooth need to divide into 9
            rl = torch.where(torch.le(regression_delta, 1.0 / 9.0), 0.5 * torch.pow(regression_delta, 2), regression_delta - 0.5 / 9.0)
            regression_loss.append(rl.mean())
        else:
            regression_loss.append(torch.tensor(0).float().cuda())

    return torch.stack(classification_loss).mean(dim=0, keepdim=True), torch.stack(regression_loss).mean(dim=0, keepdim=True)