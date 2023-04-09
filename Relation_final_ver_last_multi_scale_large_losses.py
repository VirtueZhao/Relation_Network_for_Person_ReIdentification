import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from resnet import resnet50

class RelationModel(nn.Module):
    def __init__(
        self,
        last_conv_stride=1,
        last_conv_dilation=1,
        num_stripes=6,
        local_conv_out_channels=256,
        num_classes=0):
        super(RelationModel, self).__init__()
        print("Build Model")
        self.base = resnet50(
            pretrained=True,
            last_conv_stride=last_conv_stride,
            last_conv_dilation=last_conv_dilation)
        self.num_stripes = num_stripes
        self.num_classes = num_classes

        self.local_6_conv_list = nn.ModuleList()
        self.local_4_conv_list = nn.ModuleList()
        self.local_2_conv_list = nn.ModuleList()
        self.rest_6_conv_list = nn.ModuleList()
        self.rest_4_conv_list = nn.ModuleList()
        self.rest_2_conv_list = nn.ModuleList()
        self.relation_6_conv_list = nn.ModuleList()
        self.relation_4_conv_list = nn.ModuleList()
        self.relation_2_conv_list = nn.ModuleList()
        self.global_6_max_conv_list = nn.ModuleList()
        self.global_4_max_conv_list = nn.ModuleList()
        self.global_2_max_conv_list = nn.ModuleList()
        self.global_6_rest_conv_list = nn.ModuleList()
        self.global_4_rest_conv_list = nn.ModuleList()
        self.global_2_rest_conv_list = nn.ModuleList()
        self.global_6_pooling_conv_list = nn.ModuleList()
        self.global_4_pooling_conv_list = nn.ModuleList()
        self.global_2_pooling_conv_list = nn.ModuleList()

        for i in range(num_stripes):
            self.local_6_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(4):
            self.local_4_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(2):
            self.local_2_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(num_stripes):
            self.rest_6_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(4):
            self.rest_4_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(2):
            self.rest_2_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        self.global_6_max_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_4_max_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_2_max_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_6_rest_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_4_rest_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_2_rest_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        for i in range(num_stripes):
            self.relation_6_conv_list.append(nn.Sequential(
                nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(4):
            self.relation_4_conv_list.append(nn.Sequential(
                nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(2):
            self.relation_2_conv_list.append(nn.Sequential(
                nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        self.global_6_pooling_conv_list.append(nn.Sequential(
            nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_4_pooling_conv_list.append(nn.Sequential(
            nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_2_pooling_conv_list.append(nn.Sequential(
            nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        if num_classes > 0:
            self.fc_local_6_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_6_list.append(fc)

            self.fc_local_4_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_4_list.append(fc)

            self.fc_local_2_list = nn.ModuleList()
            for _ in range(2):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_2_list.append(fc)

            self.fc_rest_6_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_rest_6_list.append(fc)

            self.fc_rest_4_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_rest_4_list.append(fc)

            self.fc_rest_2_list = nn.ModuleList()
            for _ in range(2):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_rest_2_list.append(fc)

            self.fc_local_rest_6_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_rest_6_list.append(fc)

            self.fc_local_rest_4_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_rest_4_list.append(fc)

            self.fc_local_rest_2_list = nn.ModuleList()
            for _ in range(2):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_rest_2_list.append(fc)

            self.fc_global_6_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_6_list.append(fc)

            self.fc_global_4_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_4_list.append(fc)

            self.fc_global_2_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_2_list.append(fc)

            self.fc_global_max_6_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_max_6_list.append(fc)

            self.fc_global_max_4_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_max_4_list.append(fc)

            self.fc_global_max_2_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_max_2_list.append(fc)

            self.fc_global_rest_6_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_rest_6_list.append(fc)

            self.fc_global_rest_4_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_rest_4_list.append(fc)

            self.fc_global_rest_2_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_rest_2_list.append(fc)

    def forward(self, x):
        print("Not Implemented Yet")