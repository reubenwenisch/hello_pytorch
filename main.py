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


def train(trainloader, net, params, train_loss, scheduler, freq = 100):
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
        scheduler.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        acc = 100*correct/total
        params['mb'].child.comment = f'Train Loss={loss.item():0.4f}\t Batch_id={i}\t Train Accuracy={acc:0.2f}'
        # if (i+1) % freq == 0:    # every 1000 mini-batches...
            # ...log the running loss
        # params['writer'].add_scalar('training loss', running_loss / freq, params['epoch'] * len(trainloader) + i)
        params['writer'].add_scalar('training loss', loss, params['epoch'] * len(trainloader) + i)
        params['writer'].add_scalar('Accuracy', acc, params['epoch'] * len(trainloader) + i)
            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
    train_loss.append(running_loss / 1000)
    return train_loss, acc

best_acc = 0  # best test accuracy


def valid(valloader, net, params, valid_loss, freq = 100):
    global best_acc
    net.eval()
    pbar_val = progress_bar(valloader,  parent=params['mb'])
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    val_loss = 0
    correct = 0
    total = 0
    # again no gradients needed
    with torch.no_grad():
        for i, data in enumerate(pbar_val):
            images, labels = data
            images, labels = np.transpose(images,(0,3,1,2)).to(device).to(device), labels.to(device)
            outputs = net(images)
            loss = params['criterion'](outputs, labels)
            val_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predictions.eq(labels).sum().item()
            params['mb'].child.comment = f'Test Loss={loss.item():0.4f} Batch_id={i} Val Accuracy={100*correct/total:0.2f}'
    valid_loss.append(val_loss / 1000)
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
    return valid_loss, acc


def cycle_lr(trainloader, net, params, scheduler, lr_find_epochs):
    # lr_find_epochs = 2
    # Make lists to capture the logs
    lr_find_loss = []
    lr_find_lr = []
    iter = 0
    smoothing = 0.05
    for i in range(lr_find_epochs):
        print("epoch {}".format(i))
        for inputs, labels in trainloader:
            # Send to device
            inputs, labels = np.transpose(inputs,(0,3,1,2)).to(device), labels.to(device)
            # Training mode and zero gradients
            net.train()
            params['optimizer'].zero_grad()
            
            # Get outputs to calc loss
            outputs = net(inputs)
            loss = params['criterion'](outputs, labels)

            # Backward pass
            loss.backward()
            params['optimizer'].step()

            # Update LR
            scheduler.step()
            lr_step = params['optimizer'].state_dict()["param_groups"][0]["lr"]
            lr_find_lr.append(lr_step)

            # smooth the loss
            if iter==0:
                lr_find_loss.append(loss)
            else:
                loss = smoothing  * loss + (1 - smoothing) * lr_find_loss[-1]
                lr_find_loss.append(loss)
            
            iter += 1
            params['writer'].add_scalar('lr_scheduler', lr_step, (iter*len(trainloader))+i)
            params['writer'].add_scalar('lr_training loss', loss, lr_step)
    return lr_find_lr, lr_find_loss
    
    # for i, data in enumerate(pbar, 0):
    #     params['mb'].child.comment = f'second bar stat'
    #     inputs, labels = data
    #     inputs, labels = np.transpose(inputs,(0,3,1,2)).to(device), labels.to(device)
    #     # zero the parameter gradients
    #     params['optimizer'].zero_grad()
    #     # forward + backward + optimize
    #     outputs = net(inputs)
    #     loss = params['criterion'](outputs, labels)
    #     loss.backward()
    #     params['optimizer'].step()
    #     running_loss += loss.item()
    #     _, predicted = outputs.max(1)
    #     total += labels.size(0)
    #     correct += predicted.eq(labels).sum().item()
    #     acc = 100*correct/total
    #     params['mb'].child.comment = f'Train Loss={loss.item():0.4f}\t Batch_id={i}\t Train Accuracy={acc:0.2f}'
    #     params['writer'].add_scalar('training loss', loss, params['epoch'] * len(trainloader) + i)
    #     params['writer'].add_scalar('lr_scheduler', scheduler.get_last_lr()[-1], params['epoch']+i)
    #     params['writer'].add_scalar('lr_training loss', loss, scheduler.get_last_lr()[-1]/1000)
    #     scheduler.step()
    #     if i > 100:
    #         break
    # # train_loss.append(running_loss / 1000)
