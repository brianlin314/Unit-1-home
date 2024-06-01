import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import random
import joblib
import math
from sklearn.metrics import roc_curve
from model import DeepAutoEncoder, SimpleMLP, SimpleCNN, CNNLSTM, SimpleLSTM, SimpleGRU, SimpleRNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# 固定亂數種子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)   

# 將 label 轉換為 embedding 向量，並且用一個 dictionary 保存 {"DDoS": [DDoS_embeddings], "Dos": [DoS_embeddings], ...}
def label_to_embeddings(labels, file_path): 
    label_embeddings_dict = {}
    for label in labels:
        file_name = os.path.join(file_path, f'{label}.npy')
        label_embeddings_dict[label] = np.load(file_name)
    return label_embeddings_dict

# 將 label 轉換為 index，並且用一個 dictionary 保存
def label_to_index(label_embeddings_dict): 
    keys = label_embeddings_dict.keys() # keys 是一個列表，["DDoS", "Dos", ...]
    key_index_dict = {key: i for i, key in enumerate(keys)} # key_index_dict 是一個字典，{"DDoS": 0, "Dos": 1, ...}
    return key_index_dict

# 將無窮大值替換為最大值和最小值
def replace_inf(df): 
    # 取得float64類型可表示的最大值和最小值
    max_float = np.finfo(np.float64).max
    min_float = np.finfo(np.float64).min
    # 僅選擇數值類型的欄位進行操作
    numeric_cols = df.select_dtypes(include=[np.number])
    # 遍歷所有數值列，分別取代正無窮大和負無窮大值
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], [max_float, min_float])
    return df

#  讀取資料，並且將資料分為 seen 和 unseen 兩部分
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

# 檢查是否有 GPU 可用
def cuda_available():
    print('cuda' if torch.cuda.is_available() else 'cpu')
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# 訓練 AutoEncoder
def train_AE(seen_df, unseen_df, num_epochs=100, batch_size=64, AE_model="DeepAutoEncoder"):
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
    if AE_model == "DeepAutoEncoder":
        model = DeepAutoEncoder(X_train.shape[1])
        criterion = nn.MSELoss()
    else:
        print("AE Model not found")
        return
    
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
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss}')
        save_model = '/SSD/p76111262/model_weight/CIC_AE.pth'
        torch.save(model, save_model)

    model.eval()
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

    anomalies = [idx for idx, error in enumerate(reconstruction_errors) if error > round_optimal_threshold]
    anomalies_unseen = [idx for idx, error in enumerate(reconstruction_errors_unseen) if error > round_optimal_threshold]
    print(f'pure_seen_test_acc: {(len(X_test)-len(anomalies)) / len(X_test)}')
    print(f'unseen_test_acc: {len(anomalies_unseen) / len(unseen_df)}')

    print("-------訓練 AutoEncoder 完成-------")
    return model, round_optimal_threshold

def train_classifier(seen_df, classify_model="SimpleCNN", seen_attack_list=None, label_embeddings_path=None, num_epochs=100):
    label_embeddings_dict = label_to_embeddings(seen_attack_list, label_embeddings_path) # label_embeeings 是一個字典，{"DDoS": [DDoS_embeddings], "DoS": [DoS_embeddings], ...} 
    key_index_dict = label_to_index(label_embeddings_dict) # key_index_dict 是一個字典，{"DDoS": 0, "PortScan": 1, ...}

    X = seen_df.drop('Label', axis=1) # 資料特徵
    y = seen_df['Label'] # 目標 label 

    # 將原先的 Label 都轉換為對應的 label embeddings，y_embeddings 裡面有一堆攻擊對應的向量
    y_embeddings = np.array([label_embeddings_dict[label] for label in y])

    if classify_model == "SimpleCNN":
        model = SimpleCNN()
        criterion = nn.CosineEmbeddingLoss()
    elif classify_model == "CNN-LSTM":
        model = CNNLSTM()
        criterion = nn.CosineEmbeddingLoss()
    elif classify_model == "SimpleLSTM":
        model = SimpleLSTM(input_size=78, hidden_size=64, num_layers=1)
        criterion = nn.CosineEmbeddingLoss()
    elif classify_model == "SimpleGRU":
        model = SimpleGRU(input_size=78, hidden_size=64, num_layers=1)
        criterion = nn.CosineEmbeddingLoss()
    elif classify_model == "SimpleRNN":
        model = SimpleRNN(input_size=78, hidden_size=64, num_layers=5)
        criterion = nn.CosineEmbeddingLoss()
    else:
        print("Classify Model not found")
        return
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train, X_test, y_train, y_test = train_test_split(X, y_embeddings, test_size=0.2, random_state=42)

    # 即 label 的 index，例如 [1, 4, 2, 3, 0, 1, 2, 3, 4, 0, ...]
    y_test_label = [] 
    for item in y_test:
        for key, value in label_embeddings_dict.items():
            if np.array_equal(value, item):
                y_test_label.append(key_index_dict[key])

    X_train_tensor = torch.tensor(X_train.to_numpy().reshape(-1, 1, 78), dtype=torch.float32)  # (batch_size, channels, features)，-1 會自動計算 batch_size
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.to_numpy().reshape(-1, 1, 78), dtype=torch.float32)

    ### Training ###
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor, torch.ones(len(y_train), dtype=torch.float32)) 
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model, '/SSD/p76111262/model_weight/CIC_classifier.pth')
    
    ### Testing ###
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        similarities = cosine_similarity(test_outputs.numpy(), list(label_embeddings_dict.values()))
        predicted_labels = np.argmax(similarities, axis=1)
        accuracy = accuracy_score(y_test_label, predicted_labels)
        print(f'Accuracy: {accuracy:.4f}')

    print("-------訓練 classifier 完成-------")
    return model

def evaluate(seen_df, unseen_df, threshold=0.0001, AE_model=None, classify_model=None, label_embeddings_path=None, picture_path=None):
    # if folder not exist, create it
    if not os.path.exists(f'/SSD/p76111262/ZSL_ConfusionMatrix/{picture_path}'):
        os.makedirs(f'/SSD/p76111262/ZSL_ConfusionMatrix/{picture_path}')
    seen_attack_list = seen_df['Label'].unique()
    unseen_attack_list = unseen_df['Label'].unique()
    num = len(seen_attack_list)

    seen_label_embeddings_dict = label_to_embeddings(seen_attack_list, label_embeddings_path)  
    seen_key_index_dict = label_to_index(seen_label_embeddings_dict) 
    print("seen_key_index_dict: ", seen_key_index_dict)

    unseen_label_embeddings_dict = label_to_embeddings(unseen_attack_list, label_embeddings_path) 
    unseen_key_index_dict = label_to_index(unseen_label_embeddings_dict) 
    unseen_key_index_dict = {key: value + num for key, value in unseen_key_index_dict.items()} # unseen 的 index 從 num 開始
    print("unseen_key_index_dict: ", unseen_key_index_dict)

    all_label_embeddings_dict = {**seen_label_embeddings_dict, **unseen_label_embeddings_dict}
    label_names = list(all_label_embeddings_dict.keys())
    all_key_index_dict = {**seen_key_index_dict, **unseen_key_index_dict}
    print("label_names: ", label_names)
    print("all_key_index_dict: ", all_key_index_dict)

    X_seen = seen_df.drop('Label', axis=1)
    y_seen = seen_df['Label']
    X_unseen = unseen_df.drop('Label', axis=1)
    y_unseen = unseen_df['Label']

    y_embeddings = [] 
    for label in y_seen:
        y_embeddings.append(seen_label_embeddings_dict[label])
    y_embeddings = np.array(y_embeddings)

    y_unseen_embeddings = []
    for label in y_unseen:
        y_unseen_embeddings.append(unseen_label_embeddings_dict[label])
    y_unseen_embeddings = np.array(y_unseen_embeddings)

    _, X_test, _, y_test = train_test_split(X_seen, y_embeddings, test_size=0.2, random_state=42)
    X_test = np.concatenate([X_test, X_unseen], axis=0)
    y_test = np.concatenate([y_test, y_unseen_embeddings], axis=0)

    y_test_label = [] 
    for item in y_test:
        for key, value in all_label_embeddings_dict.items():
            if np.array_equal(value, item):
                y_test_label.append(all_key_index_dict[key])

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    ##### testing ####
    AE_model.to(device)
    classify_model.to(device)
    AE_model.eval()
    classify_model.eval()
    with torch.no_grad():
        predicted_label_list = []
        for data in X_test_tensor:
            data = data.unsqueeze(0).to(device)
            AE_outputs = AE_model(data)

            error = torch.mean(torch.square(AE_outputs - data), dim=1)
            if error > threshold:
                data = data.reshape(-1, 1, 78)
                classify_output = classify_model(data)
                similarities = cosine_similarity(classify_output.cpu().numpy(), list(unseen_label_embeddings_dict.values()))
                predicted_label = np.argmax(similarities)
                predicted_label_list.append(predicted_label + num)
            else:
                data = data.reshape(-1, 1, 78)
                classify_output = classify_model(data)
                similarities = cosine_similarity(classify_output.cpu().numpy(), list(seen_label_embeddings_dict.values()))
                predicted_label = np.argmax(similarities)
                predicted_label_list.append(predicted_label)
        
        print("y_test_label: ", y_test_label)
        print("predicted_label_list: ", predicted_label_list)
        accuracy = accuracy_score(y_test_label, predicted_label_list)
        precision = precision_score(y_test_label, predicted_label_list, average='weighted')
        recall = recall_score(y_test_label, predicted_label_list, average='weighted')
        f1 = f1_score(y_test_label, predicted_label_list, average='weighted')
        cm = confusion_matrix(y_test_label, predicted_label_list)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        
        # Print confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(f'/SSD/p76111262/ZSL_ConfusionMatrix/{picture_path}/{unseen_attack_list[0]}-{unseen_attack_list[1]}.png')

        return accuracy, precision, recall, f1

def Permutations(attack_types):
    combinations = list(itertools.combinations(attack_types, 2))
    result = []

    for combo in combinations:
        two_unseens = list(combo)
        seens = [attack for attack in attack_types if attack not in two_unseens]
        
        result.append({
            'unseen_attack_name': two_unseens,
            'seen_attack_name': seens
        })

    return result

set_seed(42) # 設置隨機種子
device = cuda_available()
if __name__ == '__main__':
    AE_num_epochs = 100
    AE_batch_size = 64
    classify_num_epochs = 150
    AE_model = "DeepAutoEncoder"
    classify_model = "SimpleRNN"
    csv_file_path = '/SSD/p76111262/CIC2018_csv/preprocess_attack.csv'
    label_embeddings_path = '/SSD/p76111262/label_embedding_32'

    attack_types = [
    'DDoS_LOIC-HTTP', 'DDoS_HOIC', 'DDoS_LOIC-UDP', 'DoS_SlowHTTPTest', 'DoS_Slowloris', 'DoS_Hulk', 'DoS_GoldenEye',
    'BruteForce-XSS', 'BruteForce-Web', 'SQL-Injection', 'BruteForce-SSH', 'BruteForce-FTP', 'Infiltration', 'Botnet'
    ]
    combinations = Permutations(attack_types)

    with open(f"Best_Combinations/{classify_model}_best_com.txt", 'w') as f:
        for idx, item in enumerate(combinations):
            seen_attack_name = item['seen_attack_name']
            unseen_attack_name = item['unseen_attack_name']
            print(f"unseen attacks: {item['unseen_attack_name']}")
            print(f"seen attacks: {item['seen_attack_name']}")
            f.write(f"unseen attacks: {item['unseen_attack_name']}\n")
        
            seen_attack_df, unseen_attack_df = read_data(csv_file_path, seen_attack_name, unseen_attack_name)
            seen_attack_df_AE = seen_attack_df.copy()
            seen_attack_df_eval = seen_attack_df.copy()
            unseen_attack_df_eval = unseen_attack_df.copy()

            Trained_AE_model, best_threshold = train_AE(seen_attack_df, unseen_attack_df, AE_num_epochs, AE_batch_size, AE_model=AE_model)      
            Trained_classify_model = train_classifier(seen_attack_df_AE, classify_model=classify_model, seen_attack_list=seen_attack_name, label_embeddings_path=label_embeddings_path, num_epochs=classify_num_epochs)
            accuracy, precision, recall, f1 = evaluate(seen_attack_df_eval, unseen_attack_df_eval, threshold=best_threshold, AE_model=Trained_AE_model, classify_model=Trained_classify_model, label_embeddings_path=label_embeddings_path, picture_path=classify_model)
            f.write(f"Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1:.4f}\n")