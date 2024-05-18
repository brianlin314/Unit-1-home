import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import random
import joblib

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

def label_to_embeddings(df): # 將 label 轉換為 embedding 向量
    labels = df['Label'].unique()
    label_embeddings_dict = {}
    for label in labels:
        file_name = os.path.join(label_embeddings_file_path, f'{label}.npy')
        label_embeddings_dict[label] = np.load(file_name)
    return label_embeddings_dict

def label_to_index(label_embeddings_dict): # 將 label 轉換為 index
    keys = label_embeddings_dict.keys() # keys 是一個列表，["DDoS", "PortScan", ...]
    key_index_dict = {key: i for i, key in enumerate(keys)} # key_index_dict 是一個字典，{"DDoS": 0, "PortScan": 1, ...}
    return key_index_dict

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)   

def replace_inf(df): # 將無窮大值替換為最大值和最小值
    # 取得float64類型可表示的最大值和最小值
    max_float = np.finfo(np.float64).max
    min_float = np.finfo(np.float64).min

    # 僅選擇數值類型的欄位進行操作
    numeric_cols = df.select_dtypes(include=[np.number])

    # 遍歷所有數值列，分別取代正無窮大和負無窮大值
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], [max_float, min_float])
    return df

def read_data(file_path, seen_attack_name, unseen_attack_name):
    df = pd.read_csv(file_path)
    df = df.drop(['Timestamp'], axis=1)
    df = df.fillna(0)
    df = replace_inf(df)
    # print("origin df shape:", df.shape)
    scaler = joblib.load('/SSD/p76111262/CIC2018_csv/min_max_scaler.joblib')
    df_scaled_features = scaler.transform(df.drop(columns=['Label']))
    
    df_scaled = pd.DataFrame(df_scaled_features, columns=df.columns.drop('Label'))
    df_scaled['Label'] = df['Label'].values
    
    df_seen_and_unseen = df_scaled[df_scaled['Label'].isin(seen_attack_name + unseen_attack_name)]
    df_scaled_seen = df_scaled[df_scaled['Label'].isin(seen_attack_name)]
    df_scaled_unseen = df_scaled[df_scaled['Label'].isin(unseen_attack_name)]

    print("seen + unseen 攻擊類別有：", df_seen_and_unseen['Label'].unique())
    print("seen攻擊類別有：", df_scaled_seen['Label'].unique())
    print("unseen攻擊類別有：", df_scaled_unseen['Label'].unique())

    return df_seen_and_unseen, df_scaled_seen, df_scaled_unseen

def train(seen_and_unseen_df, seen_df, unseen_df,  num_epochs):
    set_seed(42) # 設置隨機種子

    label_embeddings_dict = label_to_embeddings(seen_and_unseen_df) # label_embeeings 是一個字典，{"DDoS": [0.1, 0.2, ...], "PortScan": [0.3, 0.4, ...], ...} 
    key_index_dict = label_to_index(label_embeddings_dict) # key_index_dict 是一個字典，{"DDoS": 0, "PortScan": 1, ...}

    X = seen_df.drop('Label', axis=1) # 資料特徵
    y = seen_df['Label'] # 目標 label 

    X_unseen = unseen_df.drop('Label', axis=1) # 資料特徵
    y_unseen = unseen_df['Label'] # 目標 label

    # 將原先的 Label 都轉換為對應的 label embeddings，y_embeddings 裡面有一堆攻擊對應的向量
    y_embeddings = [] 
    for label in y:
        y_embeddings.append(label_embeddings_dict[label])
    y_embeddings = np.array(y_embeddings)

    y_unseen_embeddings = []
    for label in y_unseen:
        y_unseen_embeddings.append(label_embeddings_dict[label])
    y_unseen_embeddings = np.array(y_unseen_embeddings)

    input_size = X.shape[1] # 特徵向量的維度
    output_size = y_embeddings.shape[1]  # label embedding 的維度

    model = Model(input_size, output_size)
    # criterion = nn.MSELoss()
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train, X_test, y_train, y_test = train_test_split(X, y_embeddings, test_size=0.2, random_state=42)

    # 將 unseen attack 加入到 test data 中
    X_test = np.concatenate([X_test, X_unseen], axis=0)
    y_test = np.concatenate([y_test, y_unseen_embeddings], axis=0)   

    # 即 label 的 index，例如 [1, 4, 2, 3, 0, 1, 2, 3, 4, 0, ...]
    y_test_label = [] 
    for item in y_test:
        for key, value in label_embeddings_dict.items():
            if np.array_equal(value, item):
                y_test_label.append(key_index_dict[key])


    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    ### Training ###
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor, torch.ones(len(y_train), dtype=torch.float32))
        # loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            similarities = cosine_similarity(test_outputs.numpy(), list(label_embeddings_dict.values()))
            print(similarities)
            predicted_labels = np.argmax(similarities, axis=1)
            accuracy = np.mean(predicted_labels == y_test_label)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
        
    print("predicted_labels:", predicted_labels)
    print("y_test_label:", y_test_label)

if __name__ == '__main__':
    num_epochs = 100
    csv_file_path = '/SSD/p76111262/CIC2018_csv/preprocess_attack.csv'
    label_embeddings_file_path = '/SSD/p76111262/label_embedding_32'
    seen_attack_name = ['DoS_GoldenEye', 'DoS_Slowloris', 'DoS_SlowHTTPTest']
    unseen_attack_name = ['DoS_Hulk']

    seen_and_unseen_attack_df, seen_attack_df, unseen_attack_df = read_data(csv_file_path, seen_attack_name, unseen_attack_name)
    train(seen_and_unseen_attack_df, seen_attack_df, unseen_attack_df, num_epochs)      
