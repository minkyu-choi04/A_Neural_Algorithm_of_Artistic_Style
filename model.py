import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.utils as utils
import numpy as np
import time
import torch.nn.functional as F
from torch.nn import init
from PIL import Image
import vgg
import argparse

content_layer = ['conv3_4']
style_layer = ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']


class Style_loss(nn.Module):
    def __init__(self, f_target):
        super(Style_loss, self).__init__()
        b, c, w, h = f_target.size()
        if b != 1:
            raise RuntimeError('batch size must be 1')
        f_target_r = f_target.detach().clone().view(1, c, w*h)
        self.G = torch.bmm(f_target_r, torch.transpose(f_target_r, 1, 2))

    def forward(self, f_current):
        b, c, w, h = f_current.size()
        f_current_r = f_current.view(1, c, w*h)
        A = torch.bmm(f_current_r, torch.transpose(f_current_r, 1, 2))
        self.loss = F.mse_loss(A, self.G)/(b*c*w*h)
        return f_current

class Content_loss(nn.Module):
    def __init__(self, f_target):
        super(Content_loss, self).__init__()
        b, c, w, h = f_target.size()
        if b != 1:
            raise RuntimeError('batch size must be 1')
        self.P = f_target.detach()

    def forward(self, f_current):
        b, c, w, h = f_current.size()
        self.loss = F.mse_loss(f_current, self.P)/2.0
        return f_current

class Normalizer(nn.Module):
    def __init__(self):
        super(Normalizer, self).__init__()
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def forward(self, img):
        img = (img-self.norm_mean)/self.norm_std
        return img


def get_layer_out(net, image_content, image_style):
    content_losses = []
    style_losses = []
    normalizer = Normalizer()
    model = nn.Sequential(normalizer)
    i = 1
    l = 1
    for layer in net.features:
        if isinstance(layer, nn.Conv2d):
            n = 'conv' + str(l) + '_' + str(i)
            i += 1
        elif isinstance(layer, nn.ReLU):
            n = 'relu' + str(l) + '_' + str(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            n = 'pool' + str(l) + '_' + str(i)
            l += 1
            i = 1
        elif isinstance(layer, nn.BatchNorm2d):
            n = 'bn' + str(l) + '_' + str(i)
        else:
            raise RuntimeError('layer element not recognized: ', layer)
        model.add_module(n, layer)

        if n in content_layer:
            out_content = model(image_content)
            content_loss = Content_loss(out_content)
            model.add_module('content_loss{}_{}'.format(l, i), content_loss)
            content_losses.append(content_loss)

        if n in style_layer:
            out_style = model(image_style)
            style_loss = Style_loss(out_style)
            model.add_module('style_loss{}_{}'.format(l, i), style_loss)
            style_losses.append(style_loss)

    return model, content_losses, style_losses

def image_loader(c_n, s_n):
    img_c = transforms.functional.to_tensor(Image.open(c_n))
    c, h, w = img_c.size()
    img_s = transforms.functional.resize(Image.open(s_n), (h, w))
    img_s = transforms.functional.to_tensor(img_s)
    return img_c.unsqueeze(0).cuda(), img_s.unsqueeze(0).cuda()

