import torch
import torch.nn as nn
import math
"""
ClassificationNet:
    input:feature_map from FPN
    output:matrix in size of (map_width * map_height * anchor_num) * class_num
    purpose:get the output,which describes the object's probability of being each class,which stands in every anchor 
"""
class ClassificationNet(nn.Module):
    def __init__(self, class_num, feature_num, anchor_num=9, feature_size=256):
        super(ClassificationNet, self).__init__()

        self.num_classes = class_num
        self.num_anchors = anchor_num

        self.conv1 = nn.Conv2d(feature_num, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, anchor_num * class_num, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        #reshape the matrix in order to get output
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


"""
RegressionNet:
    input:feature_map from FPN
    output:matrix in size of (map_width * map_height * anchor_num) * 4
    purpose:get the output,which describes the error between anchor and grand truth with (x1, y1, x2, y2)
"""
class RegressionNet(nn.Module):
    def __init__(self, feature_num, anchor_num=9, feature_size=256):
        super(RegressionNet, self).__init__()

        self.conv1 = nn.Conv2d(feature_num, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, anchor_num * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # reshape the matrix in order to get output
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class SubNet(nn.Module):
    def __init__(self, class_num, feature_num):
        super(SubNet, self).__init__()
        self.classification_net = ClassificationNet(class_num, feature_num)
        self.regression_net = RegressionNet(feature_num)

        prior = 0.01

        self.classification_net.output.weight.data.fill_(0)
        self.classification_net.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regression_net.output.weight.data.fill_(0)
        self.regression_net.output.bias.data.fill_(0)

    def forward(self, feature_maps):
        classification = torch.cat([self.classification_net(feature_map) for feature_map in feature_maps], dim=1)
        regression = torch.cat([self.regression_net(feature_map) for feature_map in feature_maps], dim=1)

        return classification, regression