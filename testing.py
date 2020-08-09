import os
import argparse
import torch
from torch import nn, utils
from torchvision import datasets, transforms
from lib.models.module import get_filter
from lib.helper import ClassifyTrainer

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--model_name', type=str, default='vgg16')
parser.add_argument('--datasets', type=str, default='cifar100')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_filters', type=int, default=3)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--save_path', type=str, default='./checkpoint')
parser.add_argument('--seed', type=int, default=20145170)
parser.set_defaults(feature=True)
args = parser.parse_args()

if args.datasets == "cifar10":
    from lib.models.cifar10 import fvgg16_bn, fresnet18
else:
    from lib.models.cifar100 import fvgg16_bn, fresnet18

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

########### num filter ############
# for f in [1, 2, 3, 4, 5, 6, 7, 8]:
# args.num_filters = f

######### part of filter ##########
# test_case = [
#     ["conv", "conv", "conv"],
#     ["conv", "uniform", "uniform"],
#     ["conv", "exp", "exp"],
#     ["conv", "normal", "uniform"],
#     ["conv", "uniform", "normal"],
#     ["conv", "normal", "normal"],
#     ["normal", "normal", "normal"],
#     ["sobel", "normal", "normal"],
#     ["line", "normal", "normal"]
# ]
#
#
# for e, t, o in test_case:
# args.edge_filter_type = e
# args.texture_filter_type = t
# args.object_filter_type = o

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# dataset
if args.datasets == "cifar10":
    test_transformer = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_dataset = datasets.CIFAR10(root='../data',
                                    train=False,
                                    transform=test_transformer,
                                    download=True)

    test_loader = utils.data.DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False)

else:
    test_transformer = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

    test_dataset = datasets.CIFAR100(root='../data',
                                     train=False,
                                     transform=test_transformer,
                                     download=True)

    test_loader = utils.data.DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True)

# model
if args.model_name == 'vgg16':
    model = fvgg16_bn(kernel_size=args.kernel_size,
                      num_filters=args.num_filters).to(args.device)

elif args.model_name == 'resnet18':
    model = fresnet18(num_filters=args.num_filters).to(args.device)

model_path = f'{args.datasets}_' \
             f'{args.model_name}_' \
             f'{args.num_filters}_' \
             f'{args.kernel_size}' \
             f'_model.pth'

model.load_state_dict(torch.load(os.path.join(args.save_path, model_path)))

# cost
criterion = nn.CrossEntropyLoss().to(args.device)
test_iter = len(test_loader)

trainer = ClassifyTrainer(model,
                          criterion,
                          train_loader=None,
                          test_loader=test_loader,
                          optimizer=None,
                          scheduler=None)

best_test_acc = 0

# train
test_loss, top1_acc, top5_acc = trainer.test()

top1_acc = top1_acc / args.batch_size
top5_acc = top5_acc / args.batch_size

print(f"TEST  [Loss / Top1 Acc / Top5 Acc] : [ {test_loss} / {top1_acc} / {top5_acc}]")