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
from torchvision import transforms
from sklearn.metrics import (confusion_matrix,
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_curve, f1_score)

from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

from model import Conv2DAutoencoder, Conv2DAutoencoder_v1
from dataloaders.dataset_ZSL import VideoDataset, ImageDataset, VideoImageDataset
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

# Function to find the optimal threshold from ROC curve
def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def get_datalaoder(domain, dataset, seen_vector_map, seen_and_unseen_vector_map, seen_lists, unseen_lists, video_type, clip_len=256):
    print("video_type:", video_type)
    if video_type == "v1":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        dataset_path = "CIC-IDS2018-ZSL-v1"
    elif video_type == "v3":
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        dataset_path = "CIC-IDS2018-ZSL"

    # Load dataset
    AE_train_dataset = ImageDataset(root_dir=f'/SSD/p76111262/{dataset_path}/{domain}/train', transform=transform, unseen_class=unseen_lists)
    AE_test_dataset = ImageDataset(root_dir=f'/SSD/p76111262/{dataset_path}/{domain}/test', transform=transform, unseen_class=unseen_lists)
    print(f'AE Number of train images: {len(AE_train_dataset)}')
    print(f'AE Number of test images: {len(AE_test_dataset)}')
    AE_train_dataloader = DataLoader(AE_train_dataset, batch_size=10, shuffle=True)
    AE_test_dataloader2 = DataLoader(AE_test_dataset, batch_size=len(AE_test_dataset), shuffle=False)
    AE_test_dataloader = DataLoader(AE_test_dataset, batch_size=1, shuffle=False)

    Classify_train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=clip_len, embedding_map=seen_vector_map, seen_list=seen_lists, unseen_list=unseen_lists), batch_size=4, shuffle=True, num_workers=0)
    Classify_test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=clip_len, embedding_map=seen_and_unseen_vector_map, seen_list=seen_lists, unseen_list=unseen_lists), batch_size=1, shuffle=True, num_workers=0)
    
    if len(Classify_test_dataloader) == len(AE_test_dataloader):
        print("test dataloader correct length")
    ZSL_test_dataloader =  DataLoader(VideoImageDataset(root=f'/SSD/p76111262/{dataset_path}/{domain}/test', clip_len=clip_len, embedding_map=seen_and_unseen_vector_map, seen_list=seen_lists, unseen_list=unseen_lists, transform=transform), batch_size=1, shuffle=False, num_workers=0)
    return AE_train_dataloader, AE_test_dataloader, AE_test_dataloader2, Classify_train_dataloader, Classify_test_dataloader, ZSL_test_dataloader

# Define the test function with accuracy calculation
def AE_test(model, dataloader):  
    model.eval()
    total_loss = 0
    outputs = []
    labels_list = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for data, labels in dataloader:
            images = data.to(device)
            reconstructed_images = model(images)
            loss = criterion(reconstructed_images, images)
            total_loss += loss.item()

            reconstruction_error = torch.mean((reconstructed_images - images) ** 2, dim=[1, 2, 3]).cpu().numpy()
            outputs.extend(reconstruction_error)
            labels_list.extend(labels.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    outputs = np.array(outputs)
    labels_list = np.array(labels_list)
    optimal_threshold = find_optimal_threshold(labels_list, outputs)
    # 四捨五入到小數點第5位
    optimal_threshold = round(optimal_threshold, 5)

    predicted_labels = (outputs > optimal_threshold).astype(int)
    accuracy = np.mean(predicted_labels == labels_list)
    print(f'Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, Optimal Threshold: {optimal_threshold:.5f}')
    return optimal_threshold

def AE_train(device, dataloader, seed, video_type=None):
    set_seed(seed)
    if video_type == "v1":
        model = Conv2DAutoencoder_v1().to(device)
    elif video_type == "v3":
        model = Conv2DAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in dataloader:
            images, _ = data
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs} Loss: {running_loss/len(dataloader)}')

    return model

def get_domain_seen_unseen_classes(domain, video_type=None):
    embedding_path = '/SSD/p76111262/label_embedding_32'
    seen_vector_map = []
    unseen_vector_map = []
    seen_and_unseen_vector_map = []

    if domain == 'DoS':
        if video_type == "v3":
            classify_seed = 17
            AE_seed = 50
            dataset = 'CIC-IDS2018-ZSL-DoS'
        elif video_type == "v1":
            classify_seed = 49
            AE_seed = 15
            dataset = 'CIC-IDS2018-ZSL-v1-DoS'
        seen = ['DoS_SlowHTTPTest', 'DoS_Hulk', 'DoS_GoldenEye']
        unseen = ['DoS_Slowloris']

    elif domain == 'DDoS':
        if video_type == "v3":
            classify_seed = 17
            AE_seed = 6
            dataset = 'CIC-IDS2018-ZSL-DDoS'
        elif video_type == "v1":
            classify_seed = 45
            AE_seed = 33
            dataset = 'CIC-IDS2018-ZSL-v1-DDoS'
        seen = ['DDoS_LOIC-HTTP', 'DDoS_HOIC'] 
        unseen = ['DDoS_LOIC-UDP']

    elif domain == 'Auth':
        if video_type == "v3":
            classify_seed = 42
            AE_seed = 26
            dataset = 'CIC-IDS2018-ZSL-Auth'
        elif video_type == "v1":
            classify_seed = 42
            AE_seed = 21
            dataset = 'CIC-IDS2018-ZSL-v1-Auth'
        seen = ['BruteForce-SSH']
        unseen =  ['BruteForce-FTP']

    elif domain == 'Web':
        if video_type == "v3":
            classify_seed = 22
            AE_seed = 1
            dataset = 'CIC-IDS2018-ZSL-Web'
        elif video_type == "v1":
            classify_seed = 16
            AE_seed = 5
            dataset = 'CIC-IDS2018-ZSL-v1-Web'
        seen = ['BruteForce-XSS', 'BruteForce-Web']
        unseen = ['SQL_Injection']

    elif domain == 'Other':
        if video_type == "v3":
            classify_seed = 42
            AE_seed = 29
            dataset = 'CIC-IDS2018-ZSL-Other'
        elif video_type == "v1":
            classify_seed = 42
            AE_seed = 36
            dataset = 'CIC-IDS2018-ZSL-v1-Other'
        seen = ['Botnet']
        unseen = ['Infiltration']

    print("Domain:", domain)
    print("Seen Attack:", seen)
    print("Unseen Attack:", unseen)

    for a in seen:
        file_name = os.path.join(embedding_path, f'{a}.npy')
        seen_vector_map.append(np.load(file_name))
        seen_and_unseen_vector_map.append(np.load(file_name))

    for a in unseen:
        file_name = os.path.join(embedding_path, f'{a}.npy')
        unseen_vector_map.append(np.load(file_name))
        seen_and_unseen_vector_map.append(np.load(file_name))

    return dataset, seen, unseen, seen_vector_map, unseen_vector_map, seen_and_unseen_vector_map, classify_seed, AE_seed

def train_classify(device, dataloader, vector_lists, seed):
    set_seed(seed)  
    # 設定模型
    model = Pac3D_ZSL_model.Pac3DClassifier(layer_sizes=(2, 2, 2, 2))
    train_params = model.parameters()
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(train_params, lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # the scheduler divides the lr by 10 every 5 epochs
    model.to(device)
    criterion.to(device)

    # 載入資料集
    print('Training model on {} dataset...'.format(dataset))
    train_size = len(dataloader.dataset)
    vector_map_tensors = [torch.tensor(vector, dtype=torch.float32) for vector in vector_lists]
    vector_map_tensor = torch.stack(vector_map_tensors).to(device)

    target = torch.ones(train_size, dtype=torch.float32, device=device)
    train_losses = []

    for epoch in range(resume_epoch, nEpochs):
        # reset the running loss and corrects
        running_loss = 0.0
        running_corrects = 0.0

        # set model to train mode
        model.train()
        for inputs, embedding, label in tqdm(dataloader):
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
        print("[train] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
    
    return model

def ZSL_test(Classify_dataloader, AE_dataloader, ZSL_test_dataloader, seen_vector_map, unseen_vector_map, Classify_model, AE_model, device, optimal_threshold):
    Classify_model.eval()
    AE_model.eval()

    running_corrects = 0
    y_true = []
    y_pred = []

    assert len(Classify_dataloader) == len(AE_dataloader)

    seen_vector_map_tensor = torch.stack([torch.tensor(vector, dtype=torch.float32) for vector in seen_vector_map]).to(device)
    unseen_vector_map_tensor = torch.stack([torch.tensor(vector, dtype=torch.float32) for vector in unseen_vector_map]).to(device)

    for inputs, _, label, AE_input in ZSL_test_dataloader:
        inputs = inputs.to(device)
        label = label.to(device)
        images = AE_input.to(device)

        with torch.no_grad():
            reconstructed_images = AE_model(images)
        reconstruction_error = torch.mean((reconstructed_images - images) ** 2, dim=[1, 2, 3]).cpu().numpy()
        
        is_seen = reconstruction_error < optimal_threshold
        for i in range(inputs.size(0)):
            if is_seen[i]:
                vector_map_tensor = seen_vector_map_tensor
            else:
                vector_map_tensor = unseen_vector_map_tensor

        with torch.no_grad():
            output = Classify_model(inputs[i].unsqueeze(0))

            # 計算每個輸出與所有標籤向量之間的 cosine similarity
            similarities = F.cosine_similarity(output, vector_map_tensor, dim=1)
            pred = similarities.argmax().item()
            if not is_seen[i]:
                pred += len(seen_vector_map_tensor)    

            correct = int(pred == label[i].item())  # 这里不需要再调用 .item() 方法
            running_corrects += correct
            y_pred.append(pred)
            y_true.append(label[i].item())

    epoch_acc = running_corrects / len(Classify_dataloader.dataset)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')  

    print("[Test] Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, f1_score: {:.4f}".format(epoch_acc, precision, recall, f1))
    print("y_true:", y_true)
    print("y_pred:", y_pred)

    return y_true, y_pred
    
if "__main__" == __name__:
    device = get_device()

    # Hyperparameters
    nEpochs = 10  # Number of epochs for training
    resume_epoch = 0  # Default is 0, change if want to resume
    clip_len = 256 # frames of each video
    domain = 'Web' # DoS, DDoS, Auth, Web, Other 
    video_type = "v1"
    dataset, seen, unseen, seen_vector_map, unseen_vector_map, seen_and_unseen_vector_map, classify_seed, AE_seed = get_domain_seen_unseen_classes(domain=domain, video_type=video_type)
    print("training ZSL with dataset:", dataset)
    AE_train_dataloader, AE_test_dataloader, AE_test_dataloader2, Classify_train_dataloader, Classify_test_dataloader, ZSL_test_dataloader = get_datalaoder(domain=domain, dataset=dataset, seen_vector_map=seen_vector_map, seen_and_unseen_vector_map=seen_and_unseen_vector_map, seen_lists=seen, unseen_lists=unseen, video_type=video_type, clip_len=clip_len)
    AE_model = AE_train(device, AE_train_dataloader, seed=AE_seed, video_type=video_type)
    optimal_threshold = AE_test(model=AE_model, dataloader=AE_test_dataloader2)
    Classify_model = train_classify(device=device, dataloader=Classify_train_dataloader, vector_lists=seen_vector_map, seed=classify_seed)
    y_true, y_pred = ZSL_test(Classify_dataloader=Classify_test_dataloader, AE_dataloader=AE_test_dataloader, ZSL_test_dataloader=ZSL_test_dataloader, seen_vector_map=seen_vector_map, unseen_vector_map=unseen_vector_map, Classify_model=Classify_model, AE_model=AE_model, device=device, optimal_threshold=optimal_threshold)
    # 生成混淆矩阵
    labels = seen + unseen
    list = []
    for i in range(len(labels)):
        list.append(i)
    print("training ZSL with dataset:", dataset)
    cm = confusion_matrix(y_true, y_pred, labels=list) 
    print("Confusion Matrix:")
    print(cm)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    if video_type == "v1":
        plt.savefig(f'{domain}_ZSL_confusion_matrix_v1.png')
    elif video_type == "v3":
        plt.savefig(f'{domain}_ZSL_confusion_matrix.png')