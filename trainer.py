import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim

from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import R3D_model, C3D_model, R2Plus1D_model, Pac3D_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import random

for seed in range(51):
    print(f"---------------training_seed_{seed}---------------------")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Device being used:", device)

    ############################
    ####    Parameters      ####
    ############################
    nEpochs = 1  # Number of epochs for training
    resume_epoch = 0  # Default is 0, change if want to resume
    useTest = True # See evolution of the test set when training
    nTestInterval = 5 # Run on test set every nTestInterval epochs
    save_epoch = 10 # Store a model every save_epoch
    lr = 1e-3 # Learning rate
    clip_len = 256 # frames of each video

    ###################################
    ####    Options of Dataset     ####
    ###################################
    dataset = 'CIC-IDS2018-v3-DoS' 
    modelName = 'Pac3D' 
    saveName = modelName + '-' + dataset

    if dataset == 'CIC-IDS2018':
        num_classes = 13
    elif dataset == 'CIC-IDS2018-v1-DoS':
        num_classes = 4
    elif dataset == 'CIC-IDS2018-v2-DoS':
        num_classes = 4
    elif dataset == 'CIC-IDS2018-v3-DoS':
        num_classes = 4
    elif dataset == 'CIC-IDS2018-v3-DDoS':
        num_classes = 3
    elif dataset == 'CIC-IDS2018-v3-Auth':
        num_classes = 2
    elif dataset == 'CIC-IDS2018-v3-Web':
        num_classes = 3
    elif dataset == 'CIC-IDS2018-v3-Other':
        num_classes = 2
    else:
        print('No Dataset')
        raise NotImplementedError


    ######################################
    ####   Load model & parameters    ####
    ######################################
    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    elif modelName == 'Pac3D':
        model = Pac3D_model.Pac3DClassifier(num_classes=num_classes, layer_sizes=(2, 2))
        train_params = model.parameters()
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError


    ######################################
    ####   Load model & parameters    ####
    ######################################
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(train_params, lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # the scheduler divides the lr by 10 every 5 epochs
    model.to(device)
    criterion.to(device)

    ########################
    ####   Load Data    ####
    ########################
    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=clip_len), batch_size=4, shuffle=True, num_workers=0)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=clip_len), batch_size=4, num_workers=0)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=clip_len), batch_size=4, num_workers=0)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)


    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    y_pred = []
    y_true = []

    for epoch in range(resume_epoch, nEpochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during trainin
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                labels = labels.type(torch.LongTensor)
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                running_loss += loss.item() * inputs.size(0) 
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
                
            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()

            best_performanse_file = f"best_performance/best_performance_{dataset}.txt"
            if epoch_acc > 0.5:
                with open(best_performanse_file, 'a') as f:
                    f.write(f"Seed: {seed} Best Performance: {epoch_acc}\n")