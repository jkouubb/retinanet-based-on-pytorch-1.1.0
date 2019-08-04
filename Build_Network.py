import Anchors
import FPN
import losses
import resnet
import SubNet
import torch
import others
import torch.nn as nn
from others import nms


class RetinaNet(nn.Module):
    def __init__(self, class_num, resnet_size=101, pretrained=True, feature_num=256):
        super(RetinaNet, self).__init__()
        self.ResNet = resnet.SetUpNet(resnet_size, pretrained)
        resnet_out_size = self.ResNet.get_size()
        self.FeaturePyramidNet = FPN.FeaturePyramidNet(resnet_out_size[0], resnet_out_size[1], resnet_out_size[2])
        self.SubNet = SubNet.SubNet(class_num, feature_num)
        self.Anchors = Anchors.Anchors()
        self.generate_predict_boxes = others.generate_predict_boxes()
        self.adjust_boxes = others.adjust_boxes()

    def forward(self, input):
        if self.training:
            image, annotations = input
            C_feature_maps = self.ResNet(image)
            P_feature_maps = self.FeaturePyramidNet(C_feature_maps)
            anchors = self.Anchors(image)
            classifications, regressions = self.SubNet(P_feature_maps)

            return losses.Calcu_Loss(regressions, classifications, anchors, annotations)
        else:
            image = input
            C_feature_maps = self.ResNet(image)
            P_feature_maps = self.FeaturePyramidNet(C_feature_maps)
            anchors = self.Anchors(image)
            classifications, regressions = self.SubNet(P_feature_maps)

            predict_boxes = self.generate_predict_boxes(anchors, regressions)
            predict_boxes = self.adjust_boxes(predict_boxes, image)

            #select the most possible class for each anchor
            scores = torch.max(classifications, dim=2, keepdim=True)[0]

            #select anchor whose possibility greater than 0.5
            valid_scores = (scores>0.05)[0, :, 0]

            if valid_scores.sum() == 0:
            # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classifications = classifications[:, valid_scores, :]
            predict_boxes = predict_boxes[:, valid_scores, :]
            scores = scores[:, valid_scores, :]

            anchors_nms_idx = nms(torch.cat([predict_boxes, scores], dim=2)[0, :, :], 0.5)

            nms_scores, nms_class = classifications[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, predict_boxes[0, anchors_nms_idx, :]]
