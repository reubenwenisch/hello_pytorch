from torch.utils.tensorboard import SummaryWriter
from utils import plot_classes_preds
import torch
from torchvision.datasets import CIFAR10
from PIL import Image
from typing import Any, Callable, Optional, Tuple
# from utils import progress_bar
import numpy as np
import os
from fastprogress.fastprogress import master_bar, progress_bar

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Cifar10Dataset(CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # image = Image.fromarray(image)
        # image = np.transpose(image,(0,3,1,2))
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def train(trainloader, net, params, train_loss, freq = 999):
    print('\nEpoch: %d' % params['epoch'])
    # loop over the dataset multiple times
    net.train()
    pbar = progress_bar(trainloader,  parent=params['mb'])
    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(pbar, 0):
        params['mb'].child.comment = f'second bar stat'
        inputs, labels = data
        inputs, labels = np.transpose(inputs,(0,3,1,2)).to(device), labels.to(device)
     # zero the parameter gradients
        params['optimizer'].zero_grad()
     # forward + backward + optimize
        outputs = net(inputs)
        loss = params['criterion'](outputs, labels)
        loss.backward()
        params['optimizer'].step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        params['mb'].child.comment = f'Train Loss={loss.item()} Batch_id={i} Train Accuracy={100*correct/total:0.2f}'
        # if i % freq == freq-1:    # every XXX mini-batches...
            # ...log the running loss
            # params['writer'].add_scalar('training loss',
            #                 running_loss / freq,
            #                 params['epoch'] * len(trainloader) + i)
            # train_loss.append(running_loss / 1000)
         # random mini-batch
            # params['writer'].add_figure('predictions vs. actuals',
            #             plot_classes_preds(net, inputs, labels),
            #             global_step=params['epoch'] * len(trainloader) + i)
    print('Loss: {}'.format(running_loss))
    # print('Finished Training')
    return running_loss/1000

best_acc = 0  # best test accuracy


def test(testloader, net, params, valid_loss, freq = 999):
    global best_acc
    net.eval()
    pbar_val = progress_bar(testloader,  parent=params['mb'])
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    test_loss = 0
    correct = 0
    total = 0
    # again no gradients needed
    with torch.no_grad():
        for i, data in enumerate(pbar_val):
            images, labels = data
            images, labels = np.transpose(images,(0,3,1,2)).to(device).to(device), labels.to(device)
            outputs = net(images)
            loss = params['criterion'](outputs, labels)
            test_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predictions.eq(labels).sum().item()
            params['mb'].child.comment = f'Test Loss={loss.item():0.4f} Batch_id={i} Test Accuracy={100*correct/total:0.2f}'
            # if i % freq == freq-1:    # every XXX mini-batches...
                # ...log the running loss
                # params['writer'].add_scalar('test loss',
                #                 test_loss / 1000,
                #                 params['epoch'] * len(testloader) + i)
                
    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / (total_pred[classname]+1)
    #     print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
    #                                                 accuracy))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': params['epoch'],
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return test_loss/1000