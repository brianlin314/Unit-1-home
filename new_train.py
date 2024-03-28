from datetime import datetime
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import R3D_model, C3D_model, R2Plus1D_model, HM3D_model, Pac3D_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

def select_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

