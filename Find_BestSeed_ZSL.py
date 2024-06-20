import glob
import os
import random
import socket
import timeit
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix,
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_curve)

from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset_ZSL import VideoDataset
from network import Pac3D_ZSL_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def get_seen_unseen_classes(domain, seed_num):
    embedding_path = '/SSD/p76111262/label_embedding_32'
    vector_map = []

    if domain == 'DoS':
        set_seed(seed_num)
        dataset = 'CIC-IDS2018-ZSL-v1-DoS'
        attack_list = ['DoS_SlowHTTPTest', 'DoS_Hulk', 'DoS_GoldenEye']

    elif domain == 'DDoS':
        set_seed(seed_num)
        dataset = 'CIC-IDS2018-ZSL-v1-DDoS'
        attack_list = ['DDoS_LOIC-HTTP', 'DDoS_HOIC'] 

    elif domain == 'Auth':
        set_seed(seed_num)
        dataset = 'CIC-IDS2018-ZSL-v1-Auth'
        attack_list = ['BruteForce-SSH']

    elif domain == 'Web':
        set_seed(seed_num)
        dataset = 'CIC-IDS2018-ZSL-v1-Web'
        attack_list = ['BruteForce-XSS', 'BruteForce-Web']

    elif domain == 'Other':
        set_seed(seed_num)
        dataset = 'CIC-IDS2018-ZSL-v1-Web'
        attack_list = ['Botnet']

    print("Domain:", domain)
    print("Attack List:", attack_list)

    for a in attack_list:
        file_name = os.path.join(embedding_path, f'{a}.npy')
        vector_map.append(np.load(file_name))

    return dataset, attack_list, vector_map

def train_classify(seed_num, device, dataset, vector_lists, attack_lists):
    # 設定模型
    model = Pac3D_ZSL_model.Pac3DClassifier(layer_sizes=(2, 2, 2, 2))
    train_params = model.parameters()
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(train_params, lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # the scheduler divides the lr by 10 every 5 epochs
    model.to(device)
    criterion.to(device)

    # 載入資料集
    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=clip_len, embedding_map=vector_lists, seen_list=attack_lists, unseen_list=None), batch_size=4, shuffle=True, num_workers=0)
    train_size = len(train_dataloader.dataset)
    vector_map_tensors = [torch.tensor(vector, dtype=torch.float32) for vector in vector_lists]
    vector_map_tensor = torch.stack(vector_map_tensors).to(device)

    target = torch.ones(train_size, dtype=torch.float32, device=device)
    train_losses = []

    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        # reset the running loss and corrects
        running_loss = 0.0
        running_corrects = 0.0

        # set model to train mode
        model.train()

        for inputs, embedding, label in tqdm(train_dataloader):
            # move inputs and labels to the device the training is taking place on
            inputs = Variable(inputs, requires_grad=True).to(device)
            embedding = Variable(embedding).to(device)
            label = Variable(label).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            batch_size = outputs.size(0)
            target = torch.ones(batch_size, device=outputs.device)
            loss = criterion(outputs, embedding, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            similarities = F.cosine_similarity(outputs.unsqueeze(1), vector_map_tensor.unsqueeze(0), dim=2)
            preds = torch.argmax(similarities, dim=1)
            running_corrects += torch.sum(preds == label)
            
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size

        train_losses.append(epoch_loss)
        stop_time = timeit.default_timer()
        print("[train] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
        print("Execution time: " + str(stop_time - start_time) + "\n")

        # 將準確率寫進 txt
        with open(f'./Best_Performance/{dataset}', 'a') as f:
            f.write(f'seed_num: {seed_num}, Loss: {epoch_loss}, Acc: {epoch_acc}\n')

if "__main__" == __name__:
    device = get_device()

    # Hyperparameters
    nEpochs = 1  # Number of epochs for training
    resume_epoch = 0  # Default is 0, change if want to resume
    save_epoch = 10 # Store a model every save_epoch
    lr = 1e-3 # Learning rate
    clip_len = 256 # frames of each video
    domain = 'DDoS' # DoS, DDoS, Auth, Web, Other 
    
    for i in range(0, 51):
        print("training_seed:", i)
        seed_num = i
        set_seed(seed_num)
        dataset, attack_list, vector_lists = get_seen_unseen_classes(domain, seed_num)
        train_classify(seed_num, device, dataset, vector_lists, attack_list)

    