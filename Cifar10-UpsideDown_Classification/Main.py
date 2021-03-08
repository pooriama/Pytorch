import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from Models import CNN
# from Datasets import Cifar100_custom,
from Datasets import CifarClassCustom
from torch.utils.data import DataLoader
import numpy as np
batch_size=8
num_epochs=10

transform =transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor()
])

transform_flip =transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor()
])

dataset = CIFAR10(root="./data",train=True,download=True)
print(dataset.data.shape)


number_of_classes=len(set(dataset.targets))
dataset1=CifarClassCustom(dataset=dataset,transform=transform,transform_flip=transform_flip)
train_data,test_data=torch.utils.data.random_split(dataset1,[40000,10000])
train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)

# train_set=Cifar100_custom(dataset,number_of_classes)
# train_data,test_data=torch.utils.data.random_split(train_set,[40000,10000])
# train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
# test_loader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)

print("number of classes",number_of_classes)
model = CNN(number_of_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion=nn.CrossEntropyLoss()



for epoch in range(num_epochs):
    train_loss=[]
    for batch_idx,(inputs,targets) in enumerate(train_loader):
        #move data to GPU
        inputs,targets=inputs.to(device),targets.to(device)


        #Forward pass
        # print("input and target shape",inputs.shape,targets.shape)
        # inputs2=np.transpose(inputs)
        # print("input and target shape", inputs.shape, targets.shape)

        outputs=model(inputs)
        loss=criterion(outputs,targets)

        train_loss.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'cost at {epoch} is {sum(train_loss)/len(train_loss)}')



#Accuracy
def accuracy_func(loader,model):
    n_correct = 0
    n_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            scores = model(x)
            # get prediction
            _, prediction = scores.max(1)

            n_correct += (prediction == y).sum()
            n_samples += prediction.size(0)
        print(f"Got {n_correct} / {n_samples} with accuracy {float(n_correct)/float(n_samples)*100}")
    model.train()

accuracy_func(train_loader,model)