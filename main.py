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

import model

def main():
    parser = argparse.ArgumentParser(description='Pytorch implementation of Neural Artistic Style Transfer')
    parser.add_argument('--w_content', default=1.0, type=float, help='Weight for content loss')
    parser.add_argument('--w_style', default=10000.0, type=float, help='Weight for style loss')
    parser.add_argument('--img_content', default='content.jpg', help='Image name for content')
    parser.add_argument('--img_style', default='style.jpg', help='Image name for style')
    parser.add_argument('--iteration', '-i', default=50, type=int, help='Total iteration')
    args = parser.parse_args()

    ### Setting parameters ###
    w_content = args.w_content
    w_style = args.w_style
    iteration = args.iteration

    ### Load Model ###
    net = vgg.vgg19(pretrained=True).cuda().eval()

    ### Load Images ###
    image_content, image_style = model.image_loader(args.img_content, args.img_style)
    image_modify = image_content.clone() 
    image_modify.requires_grad = True

    ### Iteration ###
    net_m, content_losses, style_losses = model.get_layer_out(net, image_content, image_style)
    optimi = optim.LBFGS([image_modify])
    for epoch in range(iteration):
        def closure():
            optimi.zero_grad()
            net_m(image_modify)

            content_loss_sum = 0.0
            style_loss_sum = 0.0
            for c in content_losses:
                content_loss_sum += c.loss
            for s in style_losses:
                style_loss_sum += s.loss
            loss = style_loss_sum * w_style + content_loss_sum * w_content
            loss.backward()
            if True: 
                print('epoch: {},  loss: {} / {} / {}'.format(epoch, loss.data, style_loss_sum.data*w_style, content_loss_sum.data*w_content))
            return loss
        optimi.step(closure)
        image_modify.data.clamp_(0, 1)
        utils.save_image(torch.squeeze(image_modify), 'outout{}.jpg'.format(epoch))

if __name__ == '__main__':
    main()
