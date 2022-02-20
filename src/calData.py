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
import pickle

def cal_in_cls(fold, nclasses):
    nsplit = 5
    In_classes = []
    np.random.seed(3)
    p1 = np.random.permutation(nclasses).tolist()
    nclass_split = int(nclasses/nsplit)
    Out_classes = p1[(fold - 1) * nclass_split : nclass_split * fold]
    for item in p1:
        if item not in Out_classes:
            In_classes.append(item)
    return In_classes

def testData(img, net1, criterion, noiseMagnitude1, temper, fold):
    t0 = time.time()

    norm = [63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0]
    # nclasses = int(in_dataset[5:])
    nclasses = 10
    nsplit = int(nclasses * 0.8)  # nsplit here is number of classes in ID, nclasses is total number of classes in ID
    #print("nsplit:", nsplit)

    inclass = cal_in_cls(fold, nclasses)
    in_sfx = np.array([])
    in_pro = np.array([])


    ########################################In-distribution###########################################
    images = img
    inputs = images.cuda().requires_grad_()

    outputs = net1(inputs)

    # print (inputs, outputs)
    o_output = np.zeros((images.size()[0], nclasses))

    # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
    nnOutputs = outputs.detach().cpu()
    nnOutputs = nnOutputs.numpy()

    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    for idx in range(nsplit):
        o_output[:, inclass[idx]] = nnOutputs[:, idx]
    in_sfx = np.vstack((in_sfx, o_output)) if in_sfx.size else o_output

    o_output = np.zeros((images.size()[0], nclasses))

    # Using temperature scaling
    outputs = outputs / temper

    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    maxIndexTemp = np.argmax(nnOutputs, axis=1)
    labels = torch.LongTensor(maxIndexTemp).cuda()
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    # Normalizing the gradient to the same space of image
    gradient[:, 0] = (gradient[:, 0]) / (norm[0])
    gradient[:, 1] = (gradient[:, 1]) / (norm[1])
    gradient[:, 2] = (gradient[:, 2]) / (norm[2])
    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    outputs = net1(tempInputs)
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.detach().cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    for idx in range(nsplit):
        o_output[:, inclass[idx]] = nnOutputs[:, idx]
    in_pro = np.vstack((in_pro, o_output)) if in_pro.size else o_output


    data = {'in_sfx': in_sfx, 'in_pro': in_pro}
    pickle.dump(data, open(f"./results/test_{fold}.p", "wb"))
    
    





