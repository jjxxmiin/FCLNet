import torch
import matplotlib.pyplot as plt
from torch import nn, utils
from torchvision import datasets, transforms
from lib.models.cifar10 import fvgg16_bn, fresnet18
from lib.helper import ClassifyTrainer

batch_size = 256
device = "cuda"
model_name = "vgg16"

kernel_size = 3
num_filters = 3

load_path = "./checkpoint/[].pth"

ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

origin_acc = []
origin_loss = []
prune_acc = []
prune_loss = []

for r in ratio:
    torch.manual_seed(20145170)
    torch.cuda.manual_seed(20145170)

    # augmentation
    test_transformer = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # dataset / dataloader
    test_dataset = datasets.CIFAR10(root='../data',
                                    train=False,
                                    transform=test_transformer,
                                    download=True)

    test_loader = utils.data.DataLoader(test_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

    # model
    if model_name == 'vgg16':
        model = fvgg16_bn(kernel_size=kernel_size,
                          num_filters=num_filters).to(device)

    elif model_name == 'resnet18':
        model = fresnet18(num_filters=num_filters).to(device)

    model.load_state_dict(torch.load(load_path))
    # cost
    criterion = nn.CrossEntropyLoss().to(device)

    trainer = ClassifyTrainer(model,
                              criterion,
                              train_loader=None,
                              test_loader=test_loader,
                              optimizer=None,
                              scheduler=None)

    test_loss, test_acc, _ = trainer.test()
    test_acc = test_acc / batch_size

    print(f" + ORIGIN TEST  [Loss / Acc] : [ {test_loss} / {test_acc} ]")
    origin_acc.append(test_acc)
    origin_loss.append(test_loss)

    total = 0

    start = 0

    for m in model.modules():
        if isinstance(m, nn.Conv2d) and start >= 2:
            total += m.weight.data.numel()
        start += 1

    conv_weight = torch.zeros(total).cuda()
    index = 0

    start = 0

    for m in model.modules():
        if isinstance(m, nn.Conv2d) and start >= 2:
            size = m.weight.data.numel()
            conv_weight[index:(index + size)] = m.weight.data.view(-1).abs().clone()
            index += size

        start += 1

    y, i = torch.sort(conv_weight)
    thre_index = int(total * r)
    thre = y[thre_index]

    pruned = 0
    print(f'Pruning Threshold : {thre}')
    zero_flag = False

    start = 0

    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) and start >= 2:
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print(f'layer index: {k} \t total params: {mask.numel()} \t remaining params: {int(torch.sum(mask))}')

        start += 1

    prune_trainer = ClassifyTrainer(model,
                                    criterion,
                                    train_loader=None,
                                    test_loader=test_loader,
                                    optimizer=None,
                                    scheduler=None)

    test_loss, test_acc, _ = prune_trainer.test()
    test_acc = test_acc / batch_size

    print(f" + PRUNE TEST  [Loss / Acc] : [ {test_loss} / {test_acc} ]")

    prune_acc.append(test_acc)
    prune_loss.append(test_loss)

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