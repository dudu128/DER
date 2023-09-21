import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import csv
import pandas as pd
from math import ceil,sqrt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
import copy
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, confusion_matrix ,ConfusionMatrixDisplay
import shutil
import cv2
import torchdrift
import torchvision
import pytorch_lightning as pl
import sklearn
from sklearn import manifold
from sklearn.manifold import Isomap,TSNE
import sys

repo_name = 'DER'
base_dir = os.path.realpath(".")[:os.path.realpath(".").index(repo_name) + len(repo_name)]
sys.path.insert(0, base_dir)

task_id = 2

import yaml
from inclearn.convnet import network
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import DataParallel
from easydict import EasyDict as edict


config_file = "./configs/1.yaml"
with open(config_file, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

device = "cuda:0"
# device = "cpu"

cfg = edict(config)
model = network.BasicNet(
    cfg["convnet"],
    cfg = cfg,
    nf = cfg["channel"],
    device = device,
    use_bias = cfg["use_bias"],
    dataset = cfg["dataset"],
)
parallel_model = DataParallel(model)

total_classes = 28
increments = []
increments.append(cfg["start_class"])
for _ in range((total_classes - cfg["start_class"]) // cfg["increment"]):
    increments.append(cfg["increment"])

for i in range(task_id+1):
    model.add_classes(increments[i])
    model.task_size = increments[i]

if task_id == 0:
    state_dict = torch.load(f'./ckpts2/step{task_id}.ckpt')
else:
    state_dict = torch.load(f'./ckpts2/decouple_step{task_id}.ckpt')

# parallel_model.cuda()

parallel_model.load_state_dict(state_dict)
parallel_model.eval()
# print(parallel_model)



flag = 1
DEVICE = torch.device('cuda:0')
train_set = '/hcds_vol/private/NCU/duncan/DER/imgset/oldIMG_train/'+str(count)
test_set = '/hcds_vol/private/NCU/duncan/DER/imgset/newIMG/'+str(count)



# save_path = "/hcds_vol/private/NCU/duncan/DER/isomap_draw/" +str(flag)+".png"

r_size = 256
c_crop = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
class OurDataModule(pl.LightningDataModule):
    def __init__(self, parent=None, additional_transform=None):
        if parent is None:
            self.train_dataset = torchvision.datasets.ImageFolder(train_set,
                                                                  transform=train_transform)
            self.val_dataset = torchvision.datasets.ImageFolder(test_set,
                                                                  transform=val_transform)
            self.test_dataset = torchvision.datasets.ImageFolder(test_set,
                                                                  transform=val_transform)

            self.train_batch_size = 16
            self.val_batch_size = 16
            self.additional_transform = None
            self.prepare_data_per_node = True
        else:
            self.train_dataset = parent.train_dataset
            self.val_dataset = parent.val_dataset
            self.test_dataset = parent.test_dataset
            self.train_batch_size = parent.train_batch_size
            self.val_batch_size = parent.val_batch_size
            self.additional_transform = additional_transform
        if additional_transform is not None:
            self.additional_transform = additional_transform

        self.prepare_data()
        self.setup('fit')
        self.setup('test')

    def setup(self, typ):
        pass

    def collate_fn(self, batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)
        if self.additional_transform:
            batch = (self.additional_transform(batch[0]), *batch[1:])
        return batch

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                                           num_workers=24, shuffle=True, collate_fn=self.collate_fn)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.val_batch_size,
                                           num_workers=24, shuffle=False, collate_fn=self.collate_fn)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.val_batch_size,
                                           num_workers=24, shuffle=False, collate_fn=self.collate_fn)
    def default_dataloader(self, batch_size=None, num_samples=None, shuffle=True):
        dataset = self.val_dataset
        if batch_size is None:
            batch_size = self.val_batch_size
        replacement = num_samples is not None
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=replacement, num_samples=num_samples)
        else:
            sampler = None
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                           collate_fn=self.collate_fn)    
    def reference_dataloader(self, batch_size=None, num_samples=None, shuffle=True):
        dataset = self.train_dataset
        if batch_size is None:
            batch_size = self.val_batch_size
        replacement = num_samples is not None
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=replacement, num_samples=num_samples)
        else:
            sampler = None
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                           collate_fn=self.collate_fn)

class Classifier(pl.LightningModule):
    def __init__(self, base_classifier):
        super().__init__()
        self.backbone = base_classifier
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        # self.classifier = torch.nn.Linear(512, 2) # resnet18
        self.classifier = torch.nn.Linear(2048, 2) #resnet50

    def normalize(self, x: torch.Tensor):
        # We pull the normalization, usually done in the dataset into the model forward
        x = torchvision.transforms.functional.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x

    def forward(self, x: torch.Tensor):
        x = self.normalize(x)
        y = self.backbone(x)
        return self.classifier(y)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.max(1).indices == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.max(1).indices == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.max(1).indices == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def predict(self, batch: 16, batch_idx=None, dataloader_idx = None):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ResnetPrediction(torch.nn.Module):
    def __init__(self, model):
        super(ResnetPrediction, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['feature']


train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((r_size, r_size)),
    torchvision.transforms.CenterCrop((c_crop,c_crop)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN, STD)])

val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((r_size, r_size)),
    torchvision.transforms.CenterCrop((c_crop, c_crop)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN, STD)])

datamodule = OurDataModule()

FE = ResnetPrediction(parallel_model)

FE.to(DEVICE)
# parallel_model.to(DEVICE)

#test dataloader
dataloader = datamodule.default_dataloader(batch_size=50, shuffle=False)
with torch.no_grad():
    for batch_cnt_val, data_val in enumerate(dataloader):
        inputs,_ = data_val
        inputs = inputs.to(DEVICE)
        
        if batch_cnt_val==0:
            features = parallel_model(inputs)['feature']
            # features = FE(inputs)
        else:
            features = torch.cat([features,FE(inputs)])
            # features = torch.cat([features,parallel_model(inputs)]['feature'])

# reference dataloader
ref_dataloader = datamodule.reference_dataloader(batch_size=50, shuffle=False)
with torch.no_grad():
    for batch_cnt_val, data_val in enumerate(ref_dataloader):
        inputs,_ = data_val
        inputs = inputs.to(DEVICE)
        
        if batch_cnt_val==0:
            features_ref = parallel_model(inputs)['feature']
            # features = FE(inputs)
        else:
            features_ref = torch.cat([features_ref,FE(inputs)])
            # features = torch.cat([features,parallel_model(inputs)]['feature'])


print('features: ',len(features))
print('features_ref: ',len(features_ref))

