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

def label_to_vector(df):
    labels = df['Label'].unique()
    label_vectors = {}
    for label in labels:
        label_vectors[label] = np.load(f'/SSD/p76111262/label_embedding_32/{label}.npy')
    return label_vectors
        
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)   

def replace_inf(df):
    # 取得float64類型可表示的最大值和最小值
    max_float = np.finfo(np.float64).max
    min_float = np.finfo(np.float64).min

    # 僅選擇數值類型的欄位進行操作
    numeric_cols = df.select_dtypes(include=[np.number])

    # 遍歷所有數值列，分別取代正無窮大和負無窮大值
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf], max_float)
        df[col] = df[col].replace([-np.inf], min_float)

    # 檢查替換是否成功
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
label_embeddings = label_to_vector(df)
scaler = joblib.load('/SSD/p76111262/CIC2018_csv/min_max_scaler.joblib')
X = df.drop('Label', axis=1) # 特徵
y = df['Label'] # 目標變數
y_vectors = []
for label in y:
    if label in label_embeddings:
        y_vectors.append(label_embeddings[label])
    else:
        print(f"Warning: No vector found for label '{label}'")
        y_vectors.append(np.zeros(300)) 

y_vectors = np.array(y_vectors)
X_scaled = scaler.fit_transform(X) # 對特徵進行擬合和轉換

input_size = X_scaled.shape[1]
output_size = y_vectors.shape[1]  # 標籤向量的維度
# print("output_size:", output_size)

model = Model(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_vectors, test_size=0.2, random_state=42)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train, dtype=torch.float32))
    # loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32), torch.ones(len(y_train), dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32))
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test, dtype=torch.float32))
        similarities = cosine_similarity(test_outputs.numpy(), list(label_embeddings.values()))
        # print(similarities[0])
        predicted_labels = np.argmax(similarities, axis=1)
        # print(predicted_labels)
        accuracy = np.mean(predicted_labels == np.arange(len(y_test)))
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')