from __future__ import absolute_import
import torch
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset


__all__ = ['CIFAR10Mix', 'CIFAR100Mix']

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

class CIFAR10Mix(torchvision.datasets.CIFAR10):

    def __init__(self, root, out_path, train=False, val=False,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10Mix, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.outpath = make_dataset(out_path)
        print("out_path:", out_path) #out_path is the path to the tuning dataset
        #print("self.outpath", self.outpath)

        if val:
            #ID Data
            np.random.seed(3)
            p1 = np.random.permutation(len(self.test_data))
            self.test_data = self.test_data[p1[:1000]]
            self.test_labels = [self.test_labels[i] for i in p1.tolist()[:1000]]

            #OOD Data
            np.random.seed(3)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[:1000]]
        else:
            #ID Data
            np.random.seed(3)
            p1 = np.random.permutation(len(self.test_data))
            self.test_data = self.test_data[p1[1000:]]
            self.test_labels = [self.test_labels[i] for i in p1.tolist()[1000:]]
            #OOD Data
            np.random.seed(3)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[1000:len(p1)]]

        #self.test_data is the matrices
        print(type(self.test_data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index < len(self.test_data):
            img, target = self.test_data[index], self.test_labels[index]
            img = Image.fromarray(img)
        else:
            img_path, target = self.outpath[index - len(self.test_data)], -1
            img = pil_loader(img_path)
            img = transforms.Resize(32)(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.test_data) + len(self.outpath)


class CIFAR100Mix(torchvision.datasets.CIFAR100):

    def __init__(self, root, out_path, train=False, val=False,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100Mix, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.outpath = make_dataset(out_path)
        if val:
            #ID Data
            np.random.seed(3)
            p1 = np.random.permutation(len(self.test_data))
            self.test_data = self.test_data[p1[:1000]]
            self.test_labels = [self.test_labels[i] for i in p1.tolist()[:1000]]
            #OOD Data
            np.random.seed(3)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[:1000]]
        else:
            #ID Data
            np.random.seed(3)
            p1 = np.random.permutation(len(self.test_data))
            self.test_data = self.test_data[p1[1000:]]
            self.test_labels = [self.test_labels[i] for i in p1.tolist()[1000:]]
            #OOD Data
            np.random.seed(3)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[1000:len(p1)]]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index < len(self.test_data):
            img, target = self.test_data[index], self.test_labels[index]
            img = Image.fromarray(img)
        else:
            img_path, target = self.outpath[index - len(self.test_data)], -1    #OOD val target is -1 not 0
            img = pil_loader(img_path)
            img = transforms.Resize(32)(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.test_data) + len(self.outpath)




class PIDtrain(Dataset):
    def __init__(self, root, out_path, train=False, val=False,
                 transform=None, target_transform=None):
        onlyfiles = [f for f in os.listdir(root)]  # Get names of only the files, ignoring DS_Store and README
        onlyfiles.remove(".DS_Store")
        onlyfiles.remove("README.txt")
        self.outpath = make_dataset(out_path)   # iSUN for OOD validation
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []

        for i, classes in enumerate(onlyfiles):
            classes = os.path.join(root, classes)
            for j, images in enumerate(os.listdir(classes)):
                img = np.array(Image.open(os.path.join(classes, images)))
                self.samples.append([img, i])
            print(i,classes)
        self.samples = np.array(self.samples, dtype='object')
        np.random.seed(3)
        p = np.random.permutation(len(self.samples))  #Designate 800 images to be test data


        self.test_data = self.samples[p[:800]][:,0]
        self.test_labels = self.samples[p[:800]][:,1]
        self.train_data = self.samples[p[800:]][:,0]
        self.train_labels = self.samples[p[800:]][:,1]
        if train == False:
            self.train_data = []
            self.train_labels = []

        if val:
            #ID Data
            np.random.seed(3)                                           #1000 ID from PID, 400 OOD from iSUN
            p1 = np.random.permutation(len(self.test_data))             #400 ID from PID, 400 OOD from iSUN for val
            self.test_data = self.test_data[p1[:400]]                   #400 ID from PID, 100 OOD from PIDood
            self.test_labels = [self.test_labels[i] for i in p1.tolist()[:400]]
            #OOD Data
            np.random.seed(3)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[:400]]
        else:
            #ID Data
            np.random.seed(3)
            p1 = np.random.permutation(len(self.test_data))
            self.test_data = self.test_data[p1[400:]]
            self.test_labels = [self.test_labels[i] for i in p1.tolist()[400:]]
            #OOD Data                                                  # iSUN as OOD has a lot of samples 
            #np.random.seed(3)
            #p2 = np.random.permutation(len(self.outpath))
            #self.outpath = [self.outpath[i] for i in p2.tolist()[400:len(p1)]]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)
        else:
            if index < len(self.test_data):
                img, label = self.test_data[index], self.test_labels[index]
                img = Image.fromarray(img)
            else:
                img_path, label = self.outpath[index - len(self.test_data)], -1
                img = pil_loader(img_path)
                img = transforms.Resize(32)(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data) + len(self.outpath)


class PIDonly(Dataset):
    def __init__(self, root, val=False, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.samples = np.array(make_dataset(root))

        self.test_data = self.samples
        self.test_labels = np.ones(len(self.samples)) * -1

        # 100 lifts for val, 400 for test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, label = self.test_data[index], -1
        img = pil_loader(img_path)
        img = transforms.Resize(32)(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)

#val_dataset = PIDonly('../data/PIDood')# transform=transforms.Compose([
#                        transforms.RandomCrop(32, padding=4),
#                        transforms.RandomHorizontalFlip(),
#                        transforms.ToTensor(),
#                        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
#                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
#                    ]))
class celeba(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.samples = np.array(make_dataset(root))

        self.test_data = self.samples[:10000]
        self.test_labels = np.ones(len(self.samples)) * -1

        # All images used for testing only.

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, label = self.test_data[index], -1
        img = pil_loader(img_path)
        img = transforms.Resize((32,32))(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)

