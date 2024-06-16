import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import glob
from sklearn.metrics import roc_curve, auc, precision_score
import numpy as np
import re
from dataloaders.dataset_ZSL import ImageDataset
import random

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)

# Function to find the optimal threshold from ROC curve
def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


# Define the test function with accuracy calculation
def test(model, dataloader, domain):
    model.eval()
    total_loss = 0
    outputs = []
    labels_list = []

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
    precision = precision_score(labels_list, predicted_labels, average='binary')  # Use 'macro' or 'micro' for multi-class
    print("Precision:", precision)
    print("predictions: ", predicted_labels)    
    print("labels: ", labels_list)
    print(f'Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision}, Optimal Threshold: {optimal_threshold:.4f}')

    # 將預測結果寫入檔案
    with open(f'Best_Performance/AE_predictions_{domain}.txt', 'a') as f:
        f.write(f'{accuracy:.4f}, {precision}, {optimal_threshold:.4f}\n')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for domain in ["DoS", "DDoS", "Auth", "Web", "Other"]:
    print("finding optimal threshold for domain: ", domain) 
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    if domain == 'DoS':
        unseen_class = ['DoS_Slowloris']

    elif domain == 'DDoS':
        unseen_class = ['DDoS_LOIC-UDP']

    elif domain == 'Auth':
        unseen_class =  ['BruteForce-FTP']

    elif domain == 'Web':
        unseen_class = ['SQL_Injection']

    elif domain == 'Other':
        unseen_class = ['Infiltration']

    for i in range(0, 51):
        set_seed(i)
        # Load dataset
        dataset = ImageDataset(root_dir=f'/SSD/p76111262/CIC-IDS2018-ZSL/{domain}/train', transform=transform, unseen_class=unseen_class)
        test_dataset = ImageDataset(root_dir=f'/SSD/p76111262/CIC-IDS2018-ZSL/{domain}/test', transform=transform, unseen_class=unseen_class)
        print(f'Number of train images: {len(dataset)}')
        print(f'Number of test images: {len(test_dataset)}')
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        # Model
        model = Autoencoder().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        num_epochs = 30
        for epoch in range(num_epochs):
            for data, _ in dataloader:
                img = data.to(device)
                # Forward pass
                output = model(img)
                loss = criterion(output, img)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        test(model, test_dataloader, domain=domain)
