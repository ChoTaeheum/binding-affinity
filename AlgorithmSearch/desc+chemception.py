#%% library import
import numpy as np
import pandas as pd
import torch as tc
import torch
import pickle
import time

from torch.autograd import Variable
from sklearn.utils import shuffle
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot as plt
from matplotlib import image as img
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial

#%% mini_dataset loading
with open("datasets/trainset_cnn+desc.txt", "rb") as fp:
    trainset = pickle.load(fp)

with open("datasets/validset_cnn+desc.txt", "rb") as fp:
    validset = pickle.load(fp)
    
#%% Make collate func.
def collate(samples):
    # The input `samples` is a list of pairs [(graph, label),(graph, label)].
    descs, images, labels = map(list, zip(*samples))
    return tc.tensor(descs, dtype=tc.float).cuda(), tc.tensor(images, dtype=tc.float).cuda(), tc.tensor(labels).cuda()

#%% network module 선언
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out
    

class InceptionResnet_Ablock(nn.Module):
    def __init__(self, scale=1.0):
        super(InceptionResnet_Ablock, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(4, 4, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(4, 4, kernel_size=1, stride=1),
            BasicConv2d(4, 4, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(4, 4, kernel_size=1, stride=1),
            BasicConv2d(4, 6, kernel_size=3, stride=1, padding=1),
            BasicConv2d(6, 8, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(16, 4, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out
    

class Reduction_Ablock(nn.Module):
    def __init__(self):
        super(Reduction_Ablock, self).__init__()

        self.branch0 = BasicConv2d(4, 6, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(4, 4, kernel_size=1, stride=1),
            BasicConv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BasicConv2d(4, 6, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionResnet_Bblock(nn.Module):
    def __init__(self, scale=1.0):
        super(InceptionResnet_Bblock, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(16, 16, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(16, 16, kernel_size=1, stride=1),
            BasicConv2d(16, 20, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(20, 24, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(40, 16, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out
    

class Reduction_Bblock(nn.Module):
    def __init__(self):
        super(Reduction_Bblock, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(16, 16, kernel_size=1, stride=1),
            BasicConv2d(16, 24, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(16, 16, kernel_size=1, stride=1),
            BasicConv2d(16, 20, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(16, 16, kernel_size=1, stride=1),
            BasicConv2d(16, 18, kernel_size=(3,1), stride=1, padding=(1,0)),
            BasicConv2d(18, 20, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

    
class InceptionResnet_Cblock(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super(InceptionResnet_Cblock, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(80, 80, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(80, 80, kernel_size=1, stride=1),
            BasicConv2d(80, 93, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(93, 106, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(186, 80, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, channel):
        super(SqueezeExcitation, self).__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.ReLU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.squeeze(x).view(b, c)
        out = self.excitation(out).view(b, c, 1, 1)
        out = x * out.expand_as(x)
        return out
        

#%% learning module 선언
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()    # method 상속받고 __init__()은 여기서 하겠다.
        
        self.dslayers = nn.Sequential(     
                        nn.Linear(9577, 2048),
                        nn.BatchNorm1d(2048),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(2048, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 128),
                        )
    
        self.imlayers = nn.Sequential(
                        BasicConv2d(1, 4, kernel_size=4, stride=1),
                        InceptionResnet_Ablock(scale=0.17),
                        SqueezeExcitation(channel=4),
                        InceptionResnet_Ablock(scale=0.17),
                        SqueezeExcitation(channel=4),
                        InceptionResnet_Ablock(scale=0.17),
                        SqueezeExcitation(channel=4),
                        Reduction_Ablock(), 
                        InceptionResnet_Bblock(scale=0.10),
                        SqueezeExcitation(channel=16),
                        InceptionResnet_Bblock(scale=0.10),
                        SqueezeExcitation(channel=16),
                        InceptionResnet_Bblock(scale=0.10),
                        SqueezeExcitation(channel=16),
                        Reduction_Bblock(),
                        InceptionResnet_Cblock(scale=0.20),
                        SqueezeExcitation(channel=80),
                        InceptionResnet_Cblock(scale=0.20),
                        SqueezeExcitation(channel=80),
                        InceptionResnet_Cblock(scale=0.20),
                        SqueezeExcitation(channel=80)
                        )
        
        self.avgpool = nn.AvgPool2d(68, count_include_pad=False)
        self.regress = nn.Linear(208, 1)

    def forward(self, desc, image):
        ds_h = self.dslayers(desc)
        dim = 1
        for d in ds_h.size()[1:]:
            dim = dim * d
        ds_o = ds_h.view(-1, dim)
        
        im_h = self.imlayers(image)
        im_h = self.avgpool(im_h)
        dim = 1
        for d in im_h.size()[1:]: #16, 4, 4
            dim = dim * d
        im_h = im_h.view(-1, dim)
        
        cat = tc.cat((ds_o, im_h), 1)
       
        return self.regress(cat)
    
#%% Set hyperparameter
hp_d = {}

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 4
hp_d['num_epochs'] = 300

hp_d['init_learning_rate'] = 10 ** -3.70183
hp_d['eps'] = 10 ** -8.39981
hp_d['weight_decay'] = 10 ** -3.59967

#%% Training
tr_data_loader = DataLoader(trainset, batch_size=hp_d['batch_size'], shuffle=False, collate_fn=collate)
va_data_loader = DataLoader(validset, batch_size=hp_d['batch_size'], shuffle=False, collate_fn=collate)

model = Regressor().to(torch.device('cuda:0'))
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
loss_func = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=hp_d['init_learning_rate'], 
    weight_decay=hp_d['weight_decay'], eps=hp_d['eps'])

print('tr_var:', np.var(np.array([s[2] for s in trainset])))
print('va_var:', np.var(np.array([s[2] for s in validset])))
print('total params:', total_params)

tr_epoch_losses = []
va_epoch_losses = []

start = time.time()

for epoch in range(hp_d['num_epochs']):                           #!! epoch-loop
    # training session
    model.train()
    tr_epoch_loss = 0

    for iter, (desc, image, label) in enumerate(tr_data_loader):  #!! batch-loop
        prediction = model(desc, image).view(-1)
        loss = loss_func(prediction, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tr_epoch_loss += loss.detach().item()

    tr_epoch_loss /= (iter + 1)
    print('Training epoch {}, loss {:.4f}'.format(epoch, tr_epoch_loss))
    tr_epoch_losses.append(tr_epoch_loss)

    # validation session
    model.eval()
    va_epoch_loss = 0

    for iter, (desc, image, label) in enumerate(va_data_loader):  # batch-loop
        prediction = model(desc, image).view(-1)
        loss = loss_func(prediction, label)
        
        va_epoch_loss += loss.detach().item()
        
    va_epoch_loss /= (iter + 1)
    print('Validation epoch {}, loss {:.4f}'.format(epoch, va_epoch_loss))
    va_epoch_losses.append(va_epoch_loss)

end = time.time()
print('time elapsed:', end-start)
