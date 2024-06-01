from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from torch import nn
import os

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class AutoEncoder(nn.Module):
    def __init__(self, f_in=384, f_out=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(f_in, 100),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(100, 70),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(70, f_out)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(f_out, f_out),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(f_out, 70),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(70, f_in)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def plot_confusion_matrix(matrix, labels, foldername):
    plt.figure(figsize=(16, 12))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Similarity Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    fmt = '.2f' 
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('attack label')
    plt.xlabel('attack label')
    plt.tight_layout()
    plt.show()
    savepath = os.path.join(foldername, 'sentence_bert_matrix.png')
    plt.savefig(savepath)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# 從文本文件中讀取句子
def read_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    return [sentence.strip() for sentence in sentences]

def train_autoencoder(model, data, epochs=100):
    model.train()
    for epoch in range(epochs):
        for data_point in data:
            data_point = torch.tensor(data_point, dtype=torch.float32)  # Ensure data is in the correct format
            optimizer.zero_grad()
            reconstructed = model(data_point)
            loss = criterion(reconstructed, data_point)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def encode_data(model, data):
    model.eval()
    with torch.no_grad():
        encoded_data = model.encoder(data)
    return encoded_data

# Sentences we want to encode. Example:
sentence = read_sentences_from_file("/SSD/p76111262/label_embedding/attack_sentences.txt")

# Sentences are encoded by calling model.encode()
embedding = model.encode(sentence)
print("embedding.shape:", embedding.shape)

###########################
## Auto-Encoder training ##
###########################

sentence_embedding_dim = 384
encode_embedding_dim = 32

# Initialize the auto-encoder
autoencoder = AutoEncoder(f_in=sentence_embedding_dim, f_out=encode_embedding_dim)
# Loss function
criterion = nn.MSELoss()
# Optimizer
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

train_data = torch.tensor(embedding, dtype=torch.float32)
train_autoencoder(autoencoder, train_data)

embedding = encode_data(autoencoder, train_data)

attack_name = ['DDoS', 'DoS', 'Web', 'Authentication', 'Advanced', 'DDoS_LOIC-HTTP', 'DDoS_HOIC', 'DDoS_LOIC-UDP', 'DoS_SlowHTTPTest', 'DoS_Slowloris', 'DoS_Hulk', 
               'DoS_GoldenEye', 'BruteForce-XSS', 'BruteForce-Web', 'SQL-Injection', 
               'BruteForce-SSH', 'BruteForce-FTP', 'Infiltration', 'Botnet']

foldername = f"/SSD/p76111262/label_embedding_{encode_embedding_dim}"
if not os.path.exists(foldername):
    os.makedirs(foldername)

for i, vector in enumerate(embedding):
    file = f"{attack_name[i]}.npy"
    filename = os.path.join(foldername, file)
    print(f"attack_name: {attack_name[i]}")
    print(vector)
    np.save(filename, vector) 

similarity_list = [[0 for _ in range(19)] for _ in range(19)]

for i in range(len(sentence)):
    for j in range(i, len(sentence)):
        similarity = cosine_similarity(embedding[i], embedding[j])
        similarity_list[i][j] = similarity 
        print(f"{attack_name[i]} and {attack_name[j]} Similarity Score:", {similarity})

similarity_array = np.array([np.array(row) for row in similarity_list])
print("NumPy Array:", similarity_array)
plot_confusion_matrix(similarity_array, attack_name, foldername)
