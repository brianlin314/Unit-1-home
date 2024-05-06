import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)   

def replace_inf(df):
    max_float = np.finfo(np.float64).max
    min_float = np.finfo(np.float64).min

    numeric_cols = df.select_dtypes(include=[np.number])

    for col in numeric_cols:
        df[col] = df[col].replace([np.inf], max_float)
        df[col] = df[col].replace([-np.inf], min_float)

    if not df.select_dtypes(include=[np.number]).applymap(np.isfinite).all().all():
        print("仍存在無窮大值或NaN。")
    else:
        print("所有無窮大值已成功替換。")
        
    return df

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

set_seed(42)
df = pd.read_csv('/SSD/p76111262/CIC2018_csv/preprocess_attack.csv')
df = df.drop(['Timestamp'], axis=1)
df = df.fillna(0)
df = replace_inf(df)

label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])
num_classes = len(label_encoder.classes_)

scaler = MinMaxScaler()
X = df.drop('Label', axis=1) # 特徵
y = df['Label'] # 目標變數

X_scaled = scaler.fit_transform(X) # 對特徵進行擬合和轉換
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # 劃分資料集為訓練集和測試集

input_size = X_scaled.shape[1]
output_size = num_classes

model = Model(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test, dtype=torch.float32))
        _, predicted_labels = torch.max(test_outputs, 1)
        accuracy = torch.sum(predicted_labels == torch.tensor(y_test.values, dtype=torch.long)).item() / len(y_test)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
