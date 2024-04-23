import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloaders.dataset import VideoDataset
from network.Pac3D_model import Pac3DClassifier

def save_features(attack_type, subfolder):
    num_classes_dict = {
        "DDoS": 3, "DoS": 4, "Web": 3, "Auth": 2, "Other": 2
    }
    model_path_dict = {
        "DDoS": '/SSD/p76111262/model_weight/Pac3D_run19.pth',
        "DoS": '/SSD/p76111262/model_weight/Pac3D_run20.pth',
        "Web": '/SSD/p76111262/model_weight/Pac3D_run22.pth',
        "Auth": '/SSD/p76111262/model_weight/Pac3D_run21.pth',
        "Other": '/SSD/p76111262/model_weight/Pac3D_run23.pth'
    }
    dataset_path_dict = {
        "DDoS": "CIC-IDS2018-v3-DDoS",
        "DoS": "CIC-IDS2018-v3-DoS",
        "Web": "CIC-IDS2018-v3-Web",
        "Auth": "CIC-IDS2018-v3-Auth",
        "Other": "CIC-IDS2018-v3-Other"
    }
    
    num_classes = num_classes_dict[attack_type]
    model_path = model_path_dict[attack_type]
    dataset_path = os.path.join("/path/to/Auth", subfolder)

    model = Pac3DClassifier(num_classes, layer_sizes=(2, 2))
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.eval()
    model = model.to(device)

    dataloader = DataLoader(VideoDataset(dataset_path, split='all', clip_len=256), batch_size=1, shuffle=False)

    features = []
    with torch.no_grad():  
        for inputs, _ in dataloader:  
            inputs = inputs.to(device)
            feature = model(inputs, extract_features=True) 
            features.append(feature.cpu().numpy()) 
    
    np.save(f"/SSD/p76111262/visual_embedding/{subfolder}.npy", np.concatenate(features, axis=0))         
    return features

if __name__ == '__main__':
    attack_list = ["DDoS", "DoS", "Web", "Auth", "Other"]
    for attack_type in attack_list:
        if attack_type == "DDoS":
            subfolder = ["DDoS_HOIC", "DDoS_LOIC-HTTP", "DDoS_LOIC-UDP"]
        elif attack_type == "DoS":
            subfolder = ["DoS_GoldenEye", "DoS_Hulk", "DoS_SlowHTTPTest", "DoS_Slowloris"]
        elif attack_type == "Web":
            subfolder = ["BruteForce-Web", "BruteForce-XSS", "SQL_Injection"]
        elif attack_type == "Auth":
            subfolder = ["BruteForce-FTP", "BruteForce-SSH"]
        elif attack_type == "Other":
            subfolder = ["Infiltration", "Botnet"]
        for sub in subfolder:
            save_features(attack_type, sub)
        

    