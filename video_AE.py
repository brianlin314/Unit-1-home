import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import glob
from sklearn.metrics import roc_curve, auc
import numpy as np
import re

def remove_suffix(text):
    return re.sub(r'_\d+$', '', text)

# Dataset class to load images
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, unseen_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.unseen_class = unseen_class
        file_list = glob.glob(os.path.join(root_dir, '**', '0.png'), recursive=True)
        self.filenames = [os.path.abspath(file_path) for file_path in file_list]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        parent_dir = os.path.basename(os.path.dirname(img_name))
        parent_dir = remove_suffix(parent_dir)
        is_seen = 0 if parent_dir in self.unseen_class else 1
        return image, is_seen

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, 3, stride=2, padding=1, output_padding=1),  # 将输出通道数修改为 4
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Function to find the optimal threshold from ROC curve
def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


# Define the test function with accuracy calculation
def test(model, dataloader):
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
    print(f'Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, Optimal Threshold: {optimal_threshold:.4f}')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

unseen_class = ['DoS_Slowloris']
# Load dataset
dataset = ImageDataset(root_dir='/SSD/p76111262/CIC-IDS2018-ZSL/DoS/train', transform=transform, unseen_class=unseen_class)
test_dataset = ImageDataset(root_dir='/SSD/p76111262/CIC-IDS2018-ZSL/DoS/test', transform=transform, unseen_class=unseen_class)
print(f'Number of train images: {len(dataset)}')
print(f'Number of test images: {len(test_dataset)}')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

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

# Save the model
# torch.save(model.state_dict(), 'autoencoder.pth')
# print('Model saved!')

test(model, test_dataloader)