import torch
from lib.models.cifar10 import fvgg16_bn
from lib.models.module import FCL
import matplotlib.pyplot as plt

batch_size = 200
device = "cuda"
model_name = "vgg16"
rebuild = True
num_filters = 3
kernel_size = 3

load_path = f"./checkpoint/cifar10_vgg16_{num_filters}_3_model.pth"

# model
model = fvgg16_bn(num_filters=num_filters).to(device)
model.load_state_dict(torch.load(load_path))

for name, module in model.named_modules():
    if isinstance(module, FCL):
        filters = module.filters.cpu().detach().numpy()

        fig, ax = plt.subplots(1, num_filters)

        for n in range(num_filters):
            ax[n].imshow(filters[n])

            ax[n].set_xticks([])
            ax[n].set_yticks([])

            ax[n].set_xticklabels([])
            ax[n].set_yticklabels([])

            ax[n].set_title(name)

        plt.show()

