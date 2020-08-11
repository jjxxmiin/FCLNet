import torch
from tqdm import tqdm
from torchvision import datasets, transforms


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ClassifyTrainer:
    def __init__(self,
                 model,
                 criterion,
                 train_loader=None,
                 test_loader=None,
                 optimizer=None,
                 scheduler=None,
                 device='cuda'):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.e = 0

    def train(self):
        self.model.train()

        train_iter = len(self.train_loader)

        loss_meter = AverageMeter()
        correct_meter = AverageMeter()

        for i, (images, labels) in tqdm(enumerate(self.train_loader), total=train_iter):
            if self.scheduler is not None:
                self.scheduler(self.optimizer, i, self.e)

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            # forward
            pred = self.model(images)
            # acc
            _, predicted = torch.max(pred, 1)
            correct_meter.update((predicted == labels).sum().item())
            # loss
            loss = self.criterion(pred, labels)
            loss_meter.update(loss.item())
            # backward
            loss.backward()
            # weight update
            self.optimizer.step()

        train_loss = loss_meter.avg
        train_correct = correct_meter.avg

        self.e += 1

        return train_loss, train_correct

    def test(self):
        self.model.eval()

        test_iter = len(self.test_loader)

        loss_meter = AverageMeter()
        top1_meter = AverageMeter()
        top5_meter = AverageMeter()

        for i, (images, labels) in tqdm(enumerate(self.test_loader), total=test_iter):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # forward
            output = self.model(images)
            # acc
            # _, predicted = torch.max(output, 1)
            # correct_meter.update((predicted == labels).sum().item())
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            temp_labels = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(temp_labels).float()

            top1_meter.update(correct[:, :1].sum())
            top5_meter.update(correct[:, :5].sum())

            # loss
            loss = self.criterion(output, labels)
            loss_meter.update(loss.item())

        test_loss = loss_meter.avg
        top1_acc = top1_meter.avg
        top5_acc = top5_meter.avg

        return test_loss, top1_acc, top5_acc

    def save(self, save_path):
        print("MODEL SAVED")
        torch.save(self.model.state_dict(), save_path)


def get_cifar10_test_loader(batch_size, shuffle=False):
    # augmentation
    test_transformer = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # dataset / dataloader
    test_dataset = datasets.CIFAR10(root='../data',
                                    train=False,
                                    transform=test_transformer,
                                    download=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle)

    return test_loader


def get_cifar100_test_loader(batch_size):
    # augmentation
    test_transformer = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

    # dataset / dataloader
    test_dataset = datasets.CIFAR100(root='../data',
                                     train=False,
                                     transform=test_transformer,
                                     download=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return test_loader


def cifar10_tester(model, batch_size=200, device="cuda"):
    # get test_loader
    test_loader = get_cifar10_test_loader(batch_size)

    # cost
    criterion = torch.nn.CrossEntropyLoss().to(device)

    trainer = ClassifyTrainer(model,
                              criterion,
                              train_loader=None,
                              test_loader=test_loader,
                              optimizer=None,
                              scheduler=None)

    # train
    test_loss, top1_acc, top5_acc = trainer.test()

    top1_acc = top1_acc / batch_size
    top5_acc = top5_acc / batch_size

    print(f"TEST  [Loss / Top1 Acc / Top5 Acc] : [ {test_loss} / {top1_acc} / {top5_acc}]")

    return test_loss, top1_acc, top5_acc


def cifar100_tester(model, batch_size=200, device="cuda"):
    # get test_loader
    test_loader = get_cifar100_test_loader(batch_size)

    # cost
    criterion = torch.nn.CrossEntropyLoss().to(device)

    trainer = ClassifyTrainer(model,
                              criterion,
                              train_loader=None,
                              test_loader=test_loader,
                              optimizer=None,
                              scheduler=None)

    # train
    test_loss, top1_acc, top5_acc = trainer.test()

    top1_acc = top1_acc / batch_size
    top5_acc = top5_acc / batch_size

    print(f"TEST  [Loss / Top1 Acc / Top5 Acc] : [ {test_loss} / {top1_acc} / {top5_acc}]")

    return test_loss, top1_acc, top5_acc