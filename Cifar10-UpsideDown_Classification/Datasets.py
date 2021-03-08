import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from skimage import io
from torchvision.utils import save_image
import torch
from PIL import Image



# def Cifar100_custom(dataset,number_of_classes):
#     dataset.targets=np.zeros(len(dataset.targets))
#     for i in range(0,len(dataset.data)):
#         if i%4==0:
#             transform_flip(dataset.data[i,:, :, :])
#             dataset.targets[i]=torch.tensor(1)
#         transform(dataset.data[i, :, :, :])
#     return dataset



class CifarClassCustom(Dataset):
    def __init__(self,dataset,transform,transform_flip):
        self.transfrom=transform
        self.dataset=dataset
        self.transfrom_flip=transform_flip

    def __len__(self):
        return len(self.dataset.targets)

    def __getitem__(self, index):
        rand = np.random.permutation(4)[0]

        if rand == 3:
            img = self.transfrom_flip(self.dataset.data[index, :, :, :])
            label = 1
        else:
            img = self.transfrom(self.dataset.data[index, :, :, :])
            label = 0

        # a = self.dataset.data[index, :, :, :]
        # print("a type", type(a))
        # self.transfrom(a)
        # c=torch.from_numpy(a)
        # torch.reshape(c, [3, 32, 32])
        # print("c type",type(c))
        # print("dataset shape",self.dataset.data[index,:,:,:].shape)
        # print("dataset shape",np.transpose(self.dataset.data[index,:,:,:]).shape)
        # self.dataset.data[index,:,:,:]=np.transpose(self.dataset.data[index,:,:,:])

        return img, label



