import pandas as pd
import random
import re
from sentence_transformers import SentenceTransformer
from data_config import filepaths 
print("Loading model....")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("Model loaded!")


def read_and_process_file(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Normalize newlines to spaces and remove square brackets and numbers
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces to a single space
    text = re.sub(r"\[.*?\]|\d", "", text)  # Remove square brackets and numbers
    
    # Split into sentences (simple approach, can be improved)
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    # Filter out any empty strings that may have been created during splitting
    sentences = [sentence.strip() for sentence in sentences if sentence]

    return split_train_test(sentences, 0.25)


def split_train_test(data, test_ratio):
    # Shuffling the original data to ensure random distribution
    shuffled_data = data[:]
    random.shuffle(shuffled_data)
    
    # Calculating the split index
    split_index = int(len(shuffled_data) * (1 - test_ratio))
    
    # Splitting the array
    train = shuffled_data[:split_index]
    test = shuffled_data[split_index:]
    
    return train, test

def write_strings_to_file(file_path, string_list):
    with open(file_path, 'w') as file:
        for string in string_list:
            if len(string.split(' ')) > 3:
              file.write(string + '\n')



# main script
centroids = {} # centroids (mean pooled vectors from training data)
for path in filepaths:
    train_sentences, test_sentences = read_and_process_file(path)
    path_prefix = path.split('.txt')[0]
    name = path_prefix.split('/')[-1]
    # write text files
    write_strings_to_file(path_prefix+'_train.txt', train_sentences)
    write_strings_to_file(path_prefix+'_test.txt', test_sentences)
    
    print(f"Embedding: {name}")
    train_embeddings = [model.encode(sentence) for sentence in train_sentences]
    test_embeddings = [model.encode(sentence) for sentence in test_sentences]

    train_df = pd.DataFrame(train_embeddings)
    train_df.to_csv(path_prefix+'_train_embeddings.csv', index=False, header=False)
    test_df = pd.DataFrame(test_embeddings)
    test_df.to_csv(path_prefix+'_test_embeddings.csv', index=False, header=False)
    centroid_vec = train_df.mean(axis=0) # get 1 x n vector
    centroids[name] = centroid_vec

centroid_df = pd.DataFrame(centroids)
centroid_df.to_csv('../data/centroids.csv', index=False)
print("Embedding complete.")
