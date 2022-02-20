import numpy as np
import time
from scipy import misc
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns

diff = [0, 1]
def test(plot=False):
    global diff
    cifar = np.zeros(1)
    for fold in range(1,6):
        data = pickle.load(open(f"./results/test_{fold}.p", "rb"))
        cnt = 0
        #print("len of datainpro:",len(data[f'in_pro']))
        for i in range(len(data[f'in_pro'])):
            in_probs = data[f'in_pro'][i]        #probability values of all 8 classes

            in_probs_ = in_probs[np.nonzero(in_probs)]

            in_e = - np.sum(np.log(in_probs_) * in_probs_)

            cifar[cnt] += (np.max(in_probs) - in_e)
            cnt += 1
    return cifar[0]

