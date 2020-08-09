import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lib.models.cifar10 import fvgg16_bn
from lib.models.module import FCL

kernel_size = 3
num_filters = 3
device = "cuda"

model = fvgg16_bn(kernel_size=kernel_size,
                  num_filters=num_filters).to(device)

model.load_state_dict(torch.load("./checkpoint/[].pth"))

for m in model.modules():
    if isinstance(m, FCL):
        w = m.weights.data
        w = abs(w).sum(1).sum(1).cpu().numpy()
        w = w[np.newaxis, :]
        print(f"Shape : {w.shape}")
        plt.figure(figsize=(12, 5))
        ax = sns.heatmap(w,
                         cmap='Greys',
                         cbar=False,
                         xticklabels=False,
                         yticklabels=False,
                         annot_kws={'size':5})
        plt.show()

# # build
# for i, (name, module) in enumerate(model.features.named_modules()):
#     if isinstance(module, GFLayer):
#         current_layer += 1
#
#         in_channels = module.in_ch
#         out_channels = module.out_ch
#         groups = module.groups
#         stride = module.stride
#         padding = module.padding
#
#         if current_layer <= 8:
#             f = middle_filters
#         else:
#             f = last_filters
#
#         new_weights = f.view(1, 1, 3, 3, 3) * \
#             module.weights.view(out_channels, in_channels // groups, 3, 1, 1).repeat(1, 1, 1, 3, 3)
#
#         new_weights = new_weights.sum(2)
#
#         new_conv = torch.nn.Conv2d(in_channels=in_channels,
#                                    out_channels=out_channels,
#                                    kernel_size=3,
#                                    stride=stride,
#                                    padding=padding,
#                                    groups=groups,
#                                    bias=(module.bias is not None)).to(device)
#
#         new_conv.weight.data = new_weights
#         model.features[i-1] = new_conv


# ==================================================

# edge_filter_type = "conv"
# texture_filter_type = "conv"
# object_filter_type = "conv"
#
# first_filters = get_filter(edge_filter_type,
#                            kernel_size=kernel_size,
#                            num_filters=num_filters)
#
# middle_filters = get_filter(texture_filter_type,
#                             kernel_size=kernel_size,
#                             num_filters=num_filters)
#
# last_filters = get_filter(object_filter_type,
#                           kernel_size=kernel_size,
#                           num_filters=num_filters)
#
# filters = [first_filters, middle_filters, last_filters]
#
# model = fvgg16_bn(filters=filters, kernel_size=kernel_size).to(device)
# model.load_state_dict(torch.load("./checkpoint/cifar10_vgg16_3_conv_conv_conv_model.pth"))
#
# for m in model.modules():
#     if isinstance(m, GFLayer):
#         w = m.weights.data
#
#         print(f"Shape : {w.shape}")
#         print(f"Total Params : {w.numel()}")
#         print(f"Total Zeros : {torch.sum(w < 0.001)}")