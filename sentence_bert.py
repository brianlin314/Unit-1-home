from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def plot_confusion_matrix(matrix, labels):
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
    plt.savefig('./pictures/sentence_bert_matrix.png')

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

# Sentences we want to encode. Example:
sentence = read_sentences_from_file("/SSD/p76111262/label_embedding/attack_sentences.txt")
# Sentences are encoded by calling model.encode()
embedding = model.encode(sentence)
print(embedding.shape)
print(type(embedding))


attack_name = ['DDoS', 'DoS', 'Web', 'Auth', 'Other', 'DDoS_LOIC-HTTP', 'DDoS_HOIC', 'DDoS_LOIC-UDP', 'DoS_SlowHTTPTest', 'DoS_Slowloris', 'DoS_Hulk', 
               'DoS_GoldenEye', 'BruteForce-XSS', 'BruteForce-Web', 'SQL-Injection', 
               'BruteForce-SSH', 'BruteForce-FTP', 'Infiltration', 'Botnet']

for i, vector in enumerate(embedding):
    filename = f"/SSD/p76111262/label_embedding/{attack_name[i]}.npy"  # 生成文件名
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
plot_confusion_matrix(similarity_array, attack_name)
