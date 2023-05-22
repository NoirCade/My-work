import os
from utils import *
from custom_dataset import customDataset
import torch
from torchvision import models
from torch import optim
from timm.loss import LabelSmoothingCrossEntropy
from torch import nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

# Augmentation
train_transform = A.Compose([
    A.GaussianBlur(p=0.3),
    A.RandomShadow(p=0.5),
    A.RandomRain(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# dataset
train_dataset = customDataset('./dataset/train', transform=train_transform)
val_dataset = customDataset('./dataset/val', transform=val_transform)
test_dataset = customDataset('./dataset/test', transform=test_transform)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# model call
'''
1. resnet50
2. vgg
3. mobile-net
4. swin_t
'''
# resnet50 model
# net = models.resnet50(pretrained=True)
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, 6)
# net.to(device)

# resnet18 model
net = models.resnet18()
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 6)
# net.to(device)

# vgg16 model
# net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
# num_ftrs = net.classifier[6].in_features
# net.classifier[6] = nn.Linear(num_ftrs, 6)
# net.to(device)

# mobile-net v2
# net = models.mobilenet_v2(pretrained=True)
# num_ftrs = net.classifier[1].in_features
# net.classifier[1] = nn.Linear(num_ftrs, 6)
# net.to(device)


# criterion
criterion = LabelSmoothingCrossEntropy().to(device)

# optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# model save
save_dir = './models'
os.makedirs(save_dir, exist_ok=True)

# test model
net.load_state_dict(torch.load('./models/ResNet18_11_1000.pt', map_location=device))

if __name__ == '__main__':
    num_epochs = 30
    # train(net, criterion, train_loader, val_loader, optimizer, num_epochs, device)
    test(net, test_loader, device)
