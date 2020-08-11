import torch.nn as nn
from lib.models.module import FCL

model_cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class FVGG(nn.Module):
    def __init__(self,
                 features,
                 num_class=10):

        super(FVGG, self).__init__()

        self.features = features
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def make_layers(model_cfg, num_filters, stride, padding, bias, batch_norm=False):
    layers = []

    input_channel = 3

    for i, l in enumerate(model_cfg):
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        # layers += [nn.Conv2d(input_channel, l, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2))]
        layers += [FCL(input_channel, l, kernel_size=3, num_filters=num_filters, stride=stride, padding=padding, bias=bias)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def fvgg16_bn(num_filters, stride=1, padding=1, bias=True):
    return FVGG(make_layers(model_cfg['D'], num_filters, stride, padding, bias, batch_norm=True))