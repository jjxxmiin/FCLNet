import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.module import FCL, get_filter


class FCLNet(nn.Module):
    def __init__(self, num_filters=3):
        super(FCLNet, self).__init__()

        filters = get_filter(filter_type='normal',
                             num_filters=num_filters,
                             kernel_size=3)

        self.gf1 = FCL(3, 32, filters, stride=1, padding=1)
        self.gf2 = FCL(32, 32, filters, stride=1, padding=1)

        self.classifier = nn.Sequential(nn.Linear(1568, 512),
                                        nn.Linear(512, 10))

    def forward(self, x):
        x = self.gf1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.gf2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x