import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCL(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, num_filters, stride, padding, groups=1, bias=False):
        super(FCL, self).__init__()
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.kernel_size = kernel_size
        self.num_filters = num_filters

        self.filters = nn.Parameter(torch.Tensor(num_filters, kernel_size, kernel_size))
        self.weights = nn.Parameter(torch.Tensor(out_ch, in_ch // groups, num_filters))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n1 = self.in_ch * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n1)
        self.weights.data.uniform_(-stdv, stdv)

        n2 = self.num_filters * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n2)
        self.filters.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        f = self.filters.view(1, 1, self.num_filters, self.kernel_size, self.kernel_size) * \
            self.weights.view(self.out_ch, self.in_ch // self.groups, self.num_filters, 1, 1).repeat(1, 1, 1,
                                                                                                     self.kernel_size,
                                                                                                     self.kernel_size)
        f = f.sum(2)

        if self.bias is not None:
            output = F.conv2d(x, f, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            output = F.conv2d(x, f, stride=self.stride, padding=self.padding, groups=self.groups)

        return output


def get_filter(filter_type, num_filters, kernel_size=3, device='cuda', seed=20145170):
    if filter_type == 'uniform':
        # uniform distribution [r1, r2)
        r1 = -3
        r2 = 3
        filters = torch.autograd.Variable((r1 - r2) * torch.rand(num_filters, kernel_size, kernel_size) + r2).to(device)

    elif filter_type == 'normal':
        # normal distribution mean : 0 variance : 1
        filters = torch.autograd.Variable(torch.randn(num_filters, kernel_size, kernel_size)).to(device)

    elif filter_type == 'exp':
        np.random.seed(seed)
        filters = torch.autograd.Variable(
            torch.from_numpy(np.random.exponential(size=(num_filters, kernel_size, kernel_size))).float()).to(device)

    elif filter_type == 'sobel':
        filters = torch.autograd.Variable(torch.Tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                                        [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                                        [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]])).to(device)

    elif filter_type == 'line':
        filters = torch.autograd.Variable(torch.Tensor([[[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
                                                        [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
                                                        [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
                                                        [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]])).to(device)

    else:
        print('Conv Filter')
        filters = None

    return filters


def builder(model, num_filters=3, device="cuda"):
    for name, module in model.features.named_modules():
        if isinstance(module, FCL):
            in_channels = module.in_ch
            out_channels = module.out_ch
            kernel_size = module.kernel_size
            groups = module.groups
            stride = module.stride
            padding = module.padding

            new_weights = module.filters.view(1, 1, num_filters, kernel_size, kernel_size) * \
                          module.weights.view(out_channels, in_channels // groups, num_filters, 1, 1) \
                              .repeat(1, 1, 1, kernel_size, kernel_size)

            new_weights = new_weights.sum(2)

            new_conv = torch.nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=padding,
                                       groups=groups,
                                       bias=False).to(device)

            new_conv.weight.data = new_weights

            model.features._modules[name] = new_conv

        # if isinstance(module, nn.BatchNorm2d):
        #     model.features._modules[name].track_running_stats = False

    return model