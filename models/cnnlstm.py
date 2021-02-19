import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101, resnet50, resnet34


import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
# import mmd
import torch
import numpy as np
# from Config import bottle_neck

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out

# class ResNet(nn.Module):

#     def __init__(self, block, layers, num_classes=1000):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)

#         return x

# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model


# class CNNLSTM(nn.Module):
#     def __init__(self, num_classes=2):
#         super(CNNLSTM, self).__init__()
#         self.firstNet = resnet34(pretrained=True)
#         self.firstNet.fc = nn.Linear(512,254)
#         # self.firstNet.fc = nn.Sequential(nn.Linear(self.firstNet.fc.in_features, 254))  
# #         self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
#         # self.secondLine1 = nn.Linear(512,254)
#         self.secondLine2 = nn.Linear(254,64)
#         self.secondLine3 = nn.Linear(64,2)

#     def forward(self, x):
#         x = self.firstNet(x)
#         # x = self.secondLine1(x)
#         x = self.secondLine2(x)
#         x = self.secondLine3(x)
#         return x

# class CNNLSTM(nn.Module):
#     # def __init__(self, n_classes, batch_size, device):
#     def __init__(self, num_classes=2):
#         super(CNNLSTM, self).__init__()

#         self.batch = 1#batch_size
#         self.device = 0#device

#         # Loading a VGG16
#         vgg = resnet34(pretrained=True)#models.vgg16(pretrained=True)

#         # Removing last layer of vgg 16
#         # embed = nn.Sequential(*list(vgg.classifier.children())[:-1])
#         vgg.classifier = vgg#embed

#         # Freezing the model 3 last layers
#         for param in vgg.parameters():
#             param.requires_grad = False

#         self.embedding = vgg
#         self.gru = nn.LSTM(4096, 2048, bidirectional=True)

#         # Classification layer (*2 because bidirectionnal)
#         self.classifier = nn.Sequential(
#             nn.Linear(2048 * 2, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_classes),
#         )
    
#     def forward(self, input):
#         hidden = torch.zeros(2, self.batch , 2048).to(
#             self.device
#         )

#         c_0 = torch.zeros(self.num_layer * 2, self.batch, 2048).to(
#             self.device
#         )

#         embedded = self.simple_elementwise_apply(self.embedding, input)
#         output, hidden = self.gru(embedded, (hidden, c_0))
#         hidden = hidden[0].view(-1, 2048 * 2)

#         output = self.classifier(hidden)

#         return output

#     def simple_elementwise_apply(self, fn, packed_sequence):
#         return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)


# class CNNLSTM(nn.Module):
#     def __init__(self, num_classes=2):
#         super(CNNLSTM, self).__init__()
#         self.resnet = resnet34(pretrained=True)
#         self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 3))  
#         self.lstm = nn.LSTM(input_size=3, hidden_size=256, num_layers=2)
#         self.fc1 = nn.Linear(256, 128) # Fully connected layer
#         self.fc2 = nn.Linear(128, num_classes) # Fully connected layer
    
#     def forward(self, x_3d):
#         hidden = None
#         x_3d = x_3d.unsqueeze(0) # add
#         x_ = list()
#         # print(x_3d.size())
#         for t in range(x_3d.size(1)-1):
#             # with torch.no_grad():
#             # print(x_3d[:, t, :, :, :].shape)
#             # print(x_3d[:, t+1, :, :, :].shape)
#             new1 = torch.cat([x_3d[:, 0, :, :, :], x_3d[:, 1, :, :, :]], dim=0)
#             # new1 = torch.cat([new1, x_3d[:, 2, :, :, :]], dim=0) 
#             # x = self.resnet(x_3d[:, t, :, :, :]) # x_3d[:, t, :, :, :].shape: [1,3,224,224] 
#             x = self.resnet(new1) # x_3d[:, t, :, :, :].shape: [1,3,224,224] 
#             # x.unsqueeze(0).shape(0) = seq_len
#             out, hidden = self.lstm(x.unsqueeze(0), hidden) 
#             print(out.shape)
#             print('com')
#             exit(0)
#             x = self.fc1(out[-1, :, :])
#             x = F.relu(x)
#             x = self.fc2(x)
#             if t==0:
#                 x_ = x
#             else:
#                 x_ = torch.cat([x_, x], dim=0)
#         return x_ 

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 3))  
        self.lstm = nn.LSTM(input_size=3, hidden_size=256, num_layers=2)
        self.fc1 = nn.Linear(256, 128) # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes) # Fully connected layer
    
    def forward(self, x_3d):
        hidden = None
        x_ = list()
        for t in range(x_3d.size(1)):
            x = self.resnet(x_3d[:, t, :, :, :]) # x_3d[:, t, :, :, :].shape: [1,3,224,224] 
            # print(x.shape)
            out, hidden = self.lstm(x.unsqueeze(1), hidden) 
            # print(out.shape)
            
            x = self.fc1(out[-1, :, :])
            x = F.relu(x)
            x = self.fc2(x)
            # print(x.shape)
            # exit(0)
            if t==0:
                x_ = x
            else:
                x_ = torch.cat([x_, x], dim=0)
        return x_ 
