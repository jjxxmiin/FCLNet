import torch
from lib.models.cifar10 import fvgg16_bn
from lib.helper import cifar10_tester
from lib.models.module import builder
from lib.utils import print_model_param_nums

batch_size = 200
device = "cuda"
model_name = "vgg16"
rebuild = True
num_filters = 3

load_path = f"./checkpoint/cifar10_vgg16_{num_filters}_3_model.pth"

# model
model = fvgg16_bn(num_filters=num_filters).to(device)
model.load_state_dict(torch.load(load_path))

# print param
print_model_param_nums(model)

# build
model = builder(model, num_filters=num_filters)

# test
cifar10_tester(model, batch_size=batch_size)