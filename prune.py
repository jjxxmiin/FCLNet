import torch
import matplotlib.pyplot as plt
from torch import nn, utils
from torchvision import datasets, transforms
from lib.models.cifar10 import fvgg16_bn, fresnet18
from lib.helper import cifar10_tester
from lib.models.module import builder

batch_size = 200
device = "cuda"
model_name = "vgg16"
rebuild = True

kernel_size = 3
num_filters = 3

load_path = f"./checkpoint/cifar10_vgg16_{num_filters}_3_model.pth"

ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

origin_acc = []
origin_loss = []
prune_acc = []
prune_loss = []

for r in ratio:
    # model load
    model = fvgg16_bn(num_filters=num_filters).to(device)
    model.load_state_dict(torch.load(load_path))

    # build
    conv_model = builder(model, num_filters=num_filters)

    # cost
    criterion = nn.CrossEntropyLoss().to(device)

    # test
    test_loss, test_acc, _ = cifar10_tester(conv_model, batch_size=batch_size)

    # zero weights
    total = 0

    for m in conv_model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()

    conv_weight = torch.zeros(total).cuda()

    # generate mask
    index = 0

    for m in conv_model.modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.numel()
            conv_weight[index:(index + size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weight)
    thre_index = int(total * r)
    thre = y[thre_index]

    # prune
    pruned = 0
    print(f'Pruning Threshold : {thre}')
    zero_flag = False

    for k, m in enumerate(conv_model.modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print(f'layer index: {k} \t total params: {mask.numel()} \t remaining params: {int(torch.sum(mask))}')

    # prune test
    prune_test_loss, prune_test_acc, _ = cifar10_tester(conv_model, batch_size=batch_size)

    # log
    origin_acc.append(test_acc)
    origin_loss.append(test_loss)
    prune_acc.append(prune_test_acc)
    prune_loss.append(prune_test_loss)

    print(f'Total conv params: {total}, Pruned conv params: {pruned}, Pruned ratio: {pruned / total}')

fig, axes = plt.subplots(1, 2, figsize=(10, 10))

axes[0].plot(origin_acc, label='origin acc')
axes[0].plot(prune_acc, label='prune acc')
axes[0].set_title("ACC")
axes[0].legend()
axes[0].grid()

axes[1].plot(origin_loss, label='origin loss')
axes[1].plot(prune_loss, label='prune loss')
axes[1].set_title("Loss")
axes[1].legend()
axes[1].grid()

plt.show()
