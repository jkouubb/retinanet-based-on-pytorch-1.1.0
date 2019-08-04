import torch
import torch.nn as nn
import torch.nn.functional as functional


class FeaturePyramidNet(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(FeaturePyramidNet, self).__init__()
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        C3, C4, C5 = input

        P5_1 = self.P5_1(C5)
        P5_down = functional.interpolate(P5_1, scale_factor=2, mode='nearest')
        P5_2 = self.P5_2(P5_1)

        P4_1 = self.P4_1(C4)
        P4_1 = P5_down + P4_1
        P4_down = functional.interpolate(P4_1, scale_factor=2, mode='nearest')
        P4_2 = self.P4_2(P4_1)

        P3_1 = self.P3_1(C3)
        P3_1 = P3_1 + P4_down
        P3_2 = self.P3_2(P3_1)

        P6 = self.P6(C5)

        P7_1 = self.P7_1(P6)
        P7_2 = self.P7_2(P7_1)

        return [P3_2, P4_2, P5_2, P6, P7_2]