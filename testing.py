import os
import argparse
import torch

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--model_name', type=str, default='vgg16')
parser.add_argument('--datasets', type=str, default='cifar10')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_filters', type=int, default=1)
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

# model
if args.model_name == 'vgg16':
    model = fvgg16_bn(num_filters=args.num_filters).to(args.device)

elif args.model_name == 'resnet18':
    model = fresnet18(num_filters=args.num_filters).to(args.device)

model_path = f'{args.datasets}_' \
             f'{args.model_name}_' \
             f'{args.num_filters}_' \
             f'{args.kernel_size}' \
             f'_model.pth'

model.load_state_dict(torch.load(os.path.join(args.save_path, model_path)))

if args.datasets == 'cifar10':
    from lib.helper import cifar10_tester

    cifar10_tester(model,
                   batch_size=args.batch_size,
                   device=args.device)

elif args.datasets == 'cifar100':
    from lib.helper import cifar100_tester

    cifar100_tester(model,
                    batch_size=args.batch_size,
                    device=args.device)

else:
    ValueError("The dataset should be [cifar10] or [cifar100]")