# the classifier network
# input sample size : 12 * 12 * 2500

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import nemo

# class Classifier_Net(nn.Module):
#     def __init__(self, num_classes=9):
#         super(Classifier_Net,self).__init__()
#
#         self.encoder = nn.Sequential(
#
#             nn.Conv2d(12, 32, kernel_size=(1,16), stride=(1,4), padding=0),  # 12 * 622
#             # nn.BatchNorm2d(32),
#             # nn.BatchNorm2d(32, momentum=0.9, track_running_stats=False),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2,4)),  # 6 * 155
#
#             nn.Conv2d(32, 64, kernel_size=(1,16), stride=(1,4), padding=0),  # 6 * 34
#             # nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),  # 3 * 17
#
#             # nn.Conv2d(64, 128, kernel_size=(1,6), stride=(1,2), padding=0),        #  12 * 75
#             # # nn.BatchNorm2d(128),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2),      # 6 * 37
#             #
#             # nn.Conv2d(128, 128, kernel_size=(1,6), stride=(1,2), padding=0),  # 6 * 16
#             # # nn.BatchNorm2d(128),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2),  # 3 * 8
#
#         )
#
#         self.linearLayer = nn.Sequential(
#             nn.Linear(64*3*17, 500),
#             nn.Tanh(),
#             # nn.Sigmoid(),
#             # nn.ReLU(),
#             # nn.Linear(1000, 1000),
#             # nn.Sigmoid(),
#             nn.Linear(500, num_classes),
#             nn.Sigmoid(),
#         )
#
#     def forward(self,x):
#         x1 = self.encoder(x)
#         # print(x1.size())
#         x2 = torch.flatten(x1, 1)
#         # x2 = x1.view(x1.size(0), 64*77)
#         x3 = self.linearLayer(x2)
#
#         return x3


class Classifier_Net(nn.Module):
    def __init__(self, num_classes=9):
        super(Classifier_Net, self).__init__()
        in_c = 12
        h_c = 12
        k_size = 15
        dilation = (1, 2, 4)
        layer = []
        for i in dilation:
            layer.extend(
             [weight_norm(nn.Conv1d(in_c, h_c, kernel_size=(k_size,), stride=(1,), padding=(k_size - 1, 0), bias=False)),
             # 16 * 2 * 3000 ,
             nn.BatchNorm1d(12),
             # nn.BatchNorm2d(32, momentum=0.9, track_running_stats=False),
             nn.ReLU(),
             nn.MaxPool1d(kernel_size=1)] * 20)

        self.encoder = nn.Sequential(*layer)

        self.linearLayer = nn.Sequential(
            nn.Linear(12 * 6000, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        x1 = self.encoder(x)
        # print(x1.size())
        x2 = torch.flatten(x1, 1)
        # x2 = x1.view(x1.size(0), 64*77)
        x3 = self.linearLayer(x2)

        return x3

# class Classifier_Net(nn.Module):
#     def __init__(self, num_classes=9):
#         super(Classifier_Net,self).__init__()
#
#         self.encoder = nn.Sequential(
#
#             nn.Conv2d(12, 32, kernel_size=(1,16), stride=(1,4), padding=0),  # 12 * 622
#             nn.BatchNorm2d(32),
#             # nn.BatchNorm2d(32, momentum=0.9, track_running_stats=False),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2,4)),  # 6 * 155
#
#             nn.Conv2d(32, 64, kernel_size=(1,16), stride=(1,4), padding=0),  # 6 * 34
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),  # 3 * 17
#
#             # nn.Conv2d(64, 128, kernel_size=(1,6), stride=(1,2), padding=0),        #  12 * 75
#             # # nn.BatchNorm2d(128),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2),      # 6 * 37
#             #
#             # nn.Conv2d(128, 128, kernel_size=(1,6), stride=(1,2), padding=0),  # 6 * 16
#             # # nn.BatchNorm2d(128),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2),  # 3 * 8
#
#         )
#
#         self.linearLayer = nn.Sequential(
#             nn.Linear(64*3*17, num_classes),
#             # # nn.Tanh(),
#             # nn.Sigmoid(),
#             # nn.ReLU(),
#             # nn.Linear(1000, 1000),
#             # nn.Sigmoid(),
#             # nn.Linear(1000, num_classes),
#             nn.Sigmoid(),
#         )
#
#     def forward(self,x):
#         x1 = self.encoder(x)
#         # print(x1.size())
#         x2 = torch.flatten(x1, 1)
#         # x2 = x1.view(x1.size(0), 64*77)
#         x3 = self.linearLayer(x2)
#
#         return x3


