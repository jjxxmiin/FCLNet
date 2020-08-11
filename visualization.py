import torch
from lib.models.cifar10 import fvgg16_bn
from lib.helper import get_cifar10_test_loader
from lib.models.module import builder
from lib.utils import show_grad_cam

batch_size = 200
device = "cuda"
model_name = "vgg16"
rebuild = True
num_filters = 3
label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

load_path = f"./checkpoint/cifar10_vgg16_{num_filters}_3_model.pth"

# model
model = fvgg16_bn(num_filters=num_filters).to(device)
model.load_state_dict(torch.load(load_path))

# build
model = builder(model, num_filters=num_filters)

show_grad_cam(model, label, get_cifar10_test_loader(1, shuffle=True))