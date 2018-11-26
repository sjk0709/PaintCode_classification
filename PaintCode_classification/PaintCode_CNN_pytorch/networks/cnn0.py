#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 16:38:27 2018

@author: song
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:49:19 2018

@author: song
"""

# -*- coding: utf-8 -*-
"""
Created on Tue April 3 10:56:53 2018

Convolutional VAriational Autoencode

@author: Jaekyung Song
"""

import sys, os
sys.path.append(os.pardir)  # parent directory
#import numpy as np
#import time

import torch.nn as nn
import math





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 16:38:27 2018

@author: song
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:49:19 2018

@author: song
"""

# -*- coding: utf-8 -*-
"""
Created on Tue April 3 10:56:53 2018

Convolutional VAriational Autoencode

@author: Jaekyung Song
"""

import sys, os
sys.path.append(os.pardir)  # parent directory
#import numpy as np
#import time

import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, 
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class ConvNetJK(nn.Module):

    def __init__(self, block, layers, num_classes=10, kernel_size=7, padding=3):
        self.inplanes = 64
        super(ConvNetJK, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, stride=1, padding=padding, 
                               bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
#        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  
                
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
           
        self.conv_final = nn.Conv2d(512, 512, kernel_size=3, stride=1, 
                                    padding=1, bias=False) 
        self.bn_final = nn.BatchNorm2d(512)
        
        self.fc = nn.Linear(3*3*512 * block.expansion, num_classes) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.constant_(m.weight, 1)
#                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, isGPU=True):
        
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        
        pool1, indices1 = self.maxPool(c1)
        
        l1 = self.layer1(pool1)        
        
        pool2, indices2 = self.maxPool(l1)
        
        l2 = self.layer2(pool2)         
        
        pool3, indices3 = self.maxPool(l2)
        
        l3 = self.layer3(pool3)       
        
        pool4, indices4 = self.maxPool(l3)
        
        l4 = self.layer4(pool4)       
        
        pool5, indices5 = self.maxPool(l4)
        
        out = self.conv_final(pool5)
        out = self.bn_final(out)
        out = self.relu(out)
        
        out = out.view(out.size(0), -1) 
        out = self.fc(out)
        
        return out







############################################
def jkConv10(input_size, label_size, **kwargs):
    """Constructs a ResNet-18 model.
        
    Args:
       
    """
    model = ConvNetJK(BasicBlock, [1, 1, 1, 1], num_classes=label_size, 
                   kernel_size=7, padding=3, **kwargs)    
    return model

############################################




