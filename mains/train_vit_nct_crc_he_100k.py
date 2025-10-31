from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from vit_pytorch.vit import ViT

print(f"Torch: {torch.__version__}")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

transform = transforms.Compose([transforms.ToTensor()])

data_dir = "/media/cwseitz/HDD2/NCT-CRC-HE-100K/"
nct_crc_he_100k = datasets.ImageFolder(data_dir,transform)

train_size = int(0.8*len(nct_crc_he_100k))
valid_size = len(nct_crc_he_100k) - train_size
train_dataset, val_dataset = random_split(nct_crc_he_100k, [train_size, valid_size])
num_classes = len(nct_crc_he_100k.classes)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=1),
    'val': DataLoader(val_dataset, batch_size=50, shuffle=True, num_workers=1)
}

device = 'cpu'

model = ViT(
    image_size=224,
    patch_size=8,
    num_classes=num_classes,
    depth=3, #number of transformer blocks (layers)
    dim=128, #embedding dimension
    heads=8,
    mlp_dim=512,
    channels=3,
).to(device)

batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42
seed_everything(seed)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(dataloaders['train']):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(dataloaders['train'])
        epoch_loss += loss / len(dataloaders['train'])

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in dataloaders['val']:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(dataloaders['val'])
            epoch_val_loss += val_loss / len(dataloaders['val'])

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )





