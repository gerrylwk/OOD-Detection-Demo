import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
from . import calData as d
from . import calMetric as m
import models
from utils.loader import PIDtrain, PIDonly, celeba


from PIL import Image
import os
import pandas as pd


#loading data sets

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])

criterion = nn.CrossEntropyLoss()

def test(img, net, fold, epsilon, temperature):
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    net.cuda()
    net.eval()

    d.testData(img, net, criterion, epsilon, temperature, fold)


