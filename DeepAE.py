import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
import joblib
import math
from model import DeepAutoEncoder

from sklearn.metrics import roc_curve

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
    
    scaler = joblib.load('/SSD/p76111262/CIC2018_csv/min_max_scaler.joblib')
    df_scaled_features = scaler.transform(df.drop(columns=['Label']))
    
    df_scaled = pd.DataFrame(df_scaled_features, columns=df.columns.drop('Label'))
    df_scaled['Label'] = df['Label'].values
    
    df_scaled_seen = df_scaled[df_scaled['Label'].isin(seen_attack_name)]
    df_scaled_unseen = df_scaled[df_scaled['Label'].isin(unseen_attack_name)]

    print("seen攻擊類別有：", df_scaled_seen['Label'].unique())
    print("unseen攻擊類別有：", df_scaled_unseen['Label'].unique())

    return df_scaled_seen, df_scaled_unseen

def cuda_available():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def train(seen_df, unseen_df,  num_epochs, batch_size, evaluate=False):
    set_seed(42) # 設置隨機種子
    device = cuda_available()

    # 將 seen_df 和 unseen_df 的 Label 列轉換為 0 和 1
    seen_df['Label'] = 0 # seen 為 0
    unseen_df['Label'] = 1 # unseen 為 1

    X = seen_df.drop('Label', axis=1)
    y = seen_df['Label']
    X_unseen = unseen_df.drop('Label', axis=1)
    y_unseen = unseen_df['Label']

    # 劃分資料集為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 合并 X_test 和 X_unseen
    X_combined_test = np.concatenate([X_test, X_unseen], axis=0)
    y_combined_test = np.concatenate([y_test, y_unseen], axis=0)

    X_train = torch.tensor(X.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    X_test_mixed = torch.tensor(X_combined_test, dtype=torch.float32)
    X_test_unseen = torch.tensor(X_unseen.values, dtype=torch.float32)


    EPOCHS = num_epochs
    BATCH_SIZE = batch_size
    model = DeepAutoEncoder(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(X_test, batch_size=1, shuffle=False) # 只包含 seen data 
    test_loader_mixed = DataLoader(X_test_mixed, batch_size=1, shuffle=False) # 包含seen data 與 unseen data，用來找最佳 threshold
    test_loader_unseen = DataLoader(X_test_unseen, batch_size=1, shuffle=False) # 只包含 unseen data

    model.to(device)
    training_loss = []
    for epoch in range(EPOCHS):
        loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            loss += loss.item()
        
        training_loss.append(loss)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss}')

        save_model = 'cic_anomaly.pth'
        torch.save(model, save_model)

    if evaluate == True:
        model.eval()
        device = cuda_available()

        reconstruction_errors_mixed = []
        reconstruction_errors = []
        reconstruction_errors_unseen = []

        with torch.no_grad():
            # mixed test data，為了計算最佳 threshold
            for data in test_loader_mixed:
                data = data.to(device)
                outputs = model(data)
                error = torch.mean(torch.square(outputs - data), dim=1)
                reconstruction_errors_mixed.extend(error.cpu().numpy())

            # pure test data 
            for data in test_loader:
                data = data.to(device)
                outputs = model(data)
                error = torch.mean(torch.square(outputs - data), dim=1)
                reconstruction_errors.extend(error.cpu().numpy())

            # all unseen data
            for data in test_loader_unseen:
                data = data.to(device)
                outputs = model(data)
                error = torch.mean(torch.square(outputs - data), dim=1)
                reconstruction_errors_unseen.extend(error.cpu().numpy())
        


        fpr, tpr, thresholds = roc_curve(y_combined_test, reconstruction_errors_mixed)
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        optimal_threshold = thresholds[ix]
        round_optimal_threshold = math.floor(optimal_threshold * 1000000) / 1000000 # 取小數點後5位
        print(f"Optimal Threshold: {round_optimal_threshold}")
        print(f"Best G-Mean: {gmeans[ix]}")

        anomalies = [idx for idx, error in enumerate(reconstruction_errors) if error > round_optimal_threshold]
        anomalies_unseen = [idx for idx, error in enumerate(reconstruction_errors_unseen) if error > round_optimal_threshold]
        print(f'pure_seen_test_acc: {(len(X_test)-len(anomalies)) / len(X_test)}')
        print(f'unseen_test_acc: {len(anomalies_unseen) / len(unseen_df)}')

if __name__ == '__main__':
    num_epochs = 100
    batch_size = 64
    csv_file_path = '/SSD/p76111262/CIC2018_csv/preprocess_attack.csv'

    # # DDoS and DoS attacks
    # seen_attack_name = ['DDoS_HOIC', 'DDoS_LOIC-HTTP', 'DDoS_LOIC-UDP', 'DoS_SlowHTTPTest', 'DoS_Slowloris', 'DoS_GoldenEye']
    # unseen_attack_name = ['DoS_Hulk']

    # # Web attacks
    # seen_attack_name = ['BruteForce-XSS', 'BruteForce-Web']
    # unseen_attack_name = ['SQL-Injection']

    # System intrusion attacks
    seen_attack_name = ['BruteForce-SSH', 'Infiltration']
    unseen_attack_name = ['BruteForce-FTP', 'Botnet']

    seen_attack_df, unseen_attack_df = read_data(csv_file_path, seen_attack_name, unseen_attack_name)
    train(seen_attack_df, unseen_attack_df, num_epochs, batch_size, evaluate=True)      
