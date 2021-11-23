import torch
import torchvision
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import albumentations as A
import cv2


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## Log to Tensorboard
def log_miss_imgs(writer, images):
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)

def log_model(writer, images, net):
    writer.add_graph(net, images)
    writer.close()

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
    
# def show_grid(images, labels):
#     matplotlib_imshow(torchvision.utils.make_grid(images))
#     # print labels
#     print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def show_grid(images, labels):
    # print('labels:', labels)
    print('  '.join('%5s' % classes[j] for j in labels))
    grid = torchvision.utils.make_grid(np.transpose(images,(0,3,1,2)), nrow=4)
    plt.figure(figsize=(15,15))
    plt.imshow(np.transpose(grid, (1,2,0)))

# import os
# import sys
# import time
# TOTAL_BAR_LENGTH = 65.
# last_time = time.time()
# begin_time = last_time

# # _, term_width = os.popen('stty size', 'r').read().split()
# # term_width = int(term_width)

# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.

#     cur_len = int(TOTAL_BAR_LENGTH*current/total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

#     sys.stdout.write(' [')
#     for i in range(cur_len):
#         sys.stdout.write('=')
#     sys.stdout.write('>')
#     for i in range(rest_len):
#         sys.stdout.write('.')
#     sys.stdout.write(']')

#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time

#     L = []
#     L.append('  Step: %s' % format_time(step_time))
#     L.append(' | Tot: %s' % format_time(tot_time))
#     if msg:
#         L.append(' | ' + msg)

#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#         sys.stdout.write(' ')

#     # Go back to the center of the bar.
#     for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#         sys.stdout.write('\b')
#     sys.stdout.write(' %d/%d ' % (current+1, total))

#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()

# def format_time(seconds):
#     days = int(seconds / 3600/24)
#     seconds = seconds - days*3600*24
#     hours = int(seconds / 3600)
#     seconds = seconds - hours*3600
#     minutes = int(seconds / 60)
#     seconds = seconds - minutes*60
#     secondsf = int(seconds)
#     seconds = seconds - secondsf
#     millis = int(seconds*1000)

#     f = ''
#     i = 1
#     if days > 0:
#         f += str(days) + 'D'
#         i += 1
#     if hours > 0 and i <= 2:
#         f += str(hours) + 'h'
#         i += 1
#     if minutes > 0 and i <= 2:
#         f += str(minutes) + 'm'
#         i += 1
#     if secondsf > 0 and i <= 2:
#         f += str(secondsf) + 's'
#         i += 1
#     if millis > 0 and i <= 2:
#         f += str(millis) + 'ms'
#         i += 1
#     if f == '':
#         f = '0ms'
#     return f