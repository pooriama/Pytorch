import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,number_of_classes):
        super(CNN,self).__init__()
        #define the conv layers
        self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1)
        self.pool1=nn.MaxPool2d(kernel_size=3)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1)
        self.Batchnorm1 = nn.BatchNorm2d(32)
        self.Batchnorm2 = nn.BatchNorm2d(64)
        self.Batchnorm3 = nn.BatchNorm2d(128)

        #define the Linear Layer
        self.fc1=nn.Linear(4608,512)
        self.fc2=nn.Linear(512,number_of_classes)

    def forward(self,x):
        x=F.relu(self.Batchnorm1(self.conv1(x)))
        x=self.pool1(x)
        x=F.relu(self.Batchnorm2(self.conv2(x)))
        x=F.relu(self.Batchnorm3(self.conv3(x)))
        x=x.reshape(x.shape[0],-1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x