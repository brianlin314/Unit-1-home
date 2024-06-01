from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


nltk.download('stopwords')
nltk.download('punkt')

def load_model(model_path):
    # Load the Word2Vec model from the specified path
    model = Word2Vec.load(model_path)
    return model

def remove_stopwords_and_punctuation(words):
    stop_words = set(stopwords.words('english'))
    punct_table = str.maketrans('', '', string.punctuation)
    filtered_words = [word.lower().translate(punct_table) for word in words if word.lower().translate(punct_table) not in stop_words and word.lower().translate(punct_table) != '']
    return filtered_words

def preprocess_sentences(input_file, output_file):
    updated_sentences = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            updated_sentences.append(line.strip())

    tokenized_sentences = []
    for sentence in updated_sentences:
        words = word_tokenize(sentence)
        filtered_words = remove_stopwords_and_punctuation(words) 
        tokenized_sentences.append(filtered_words)

    with open(output_file, 'w', encoding='utf-8') as file:
        for sentence in tokenized_sentences:
            file.write(' '.join(sentence) + '\n')

def train_word2vec_model(sentences_file, model_save_path):
    # Load the updated sentences as a LineSentence object
    updated_sentences = LineSentence(sentences_file)

    # Training the Word2Vec model on the updated sentences
    updated_model = Word2Vec(sentences=updated_sentences, vector_size=300, window=10, min_count=3, workers=4)

    # Save the updated model
    updated_model.save(model_save_path)

def remove_punctuation_and_lower(text):
  punct_translation_table = str.maketrans('', '', string.punctuation)
  return text.translate(punct_translation_table).lower()

def plot_similarity_matrix(df, labels):
    plt.figure(figsize=(16, 12))
    
    # Convert DataFrame to a NumPy array for plotting
    matrix = df.values
    # Ensure the diagonal is set to 1.0 and strictly lower triangle to 0.0
    np.fill_diagonal(matrix, 1.0)
    tril_indices = np.tril_indices_from(matrix, k=-1)
    matrix[tril_indices] = 0.0
    
    # Plotting the matrix using imshow
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Word Similarity Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # Annotating the matrix with text
    fmt = '.2f' 
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, format(matrix[i, j], fmt),
                     ha="center", va="center",
                     color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('attack Label')
    plt.xlabel('attack Label')
    plt.tight_layout()
    plt.savefig("/SSD/p76111262/word2vec/word_similarity_matrix.png")

def create_similarity_matrix(words, model):
    # Initialize an empty matrix
    similarity_matrix = np.zeros((len(words), len(words)))

    # Compute similarity between each pair of words
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if word1 in model.wv and word2 in model.wv:
                similarity_matrix[i, j] = model.wv.similarity(word1, word2)
            else:
                similarity_matrix[i, j] = None  # None for any words not present in the model

    return pd.DataFrame(similarity_matrix, index=words, columns=words)

if __name__ == "__main__":
    # Example usage:
    input_file = "/SSD/p76111262/word2vec/cybersecurity_sentences.txt"
    output_file = "/SSD/p76111262/word2vec/preprocessed_cybersecurity_sentences.txt"
    model_save_path = "/SSD/p76111262/word2vec/cybersecurity_word2vec.model"

    # Preprocess sentences and save to output file
    preprocess_sentences(input_file, output_file)
    # Train Word2Vec model and save
    train_word2vec_model(output_file, model_save_path)

    words_of_interest = ['DDoS', 'DoS', 'Web', 'Authentication', 'Advanced', 'DDoS_LOIC-HTTP', 'DDoS_HOIC', 'DDoS_LOIC-UDP', 'DoS_SlowHTTPTest', 'DoS_Slowloris', 'DoS_Hulk', 
               'DoS_GoldenEye', 'BruteForce-XSS', 'BruteForce-Web', 'SQL-Injection', 
               'BruteForce-SSH', 'BruteForce-FTP', 'Infiltration', 'Botnet']
    
    words_of_interest_processed = []
    for word in words_of_interest:
        word = remove_punctuation_and_lower(word)
        words_of_interest_processed.append(word)

    word2vec_model = load_model(model_save_path)

    # Create the similarity matrix
    sim_matrix = create_similarity_matrix(words_of_interest_processed, word2vec_model)

    plot_similarity_matrix(sim_matrix, words_of_interest)