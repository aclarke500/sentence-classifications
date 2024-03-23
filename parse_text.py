import re
from sentence_transformers import SentenceTransformer
import pandas as pd

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

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

    return sentences



# Example usage
aita_path = 'AITA.txt'
attention_path = 'Attention.txt'
aita_array = read_and_process_file(aita_path)
attention_array = read_and_process_file(attention_path)
print('he')

aita_encodings = [model.encode(sentence) for sentence in aita_array]
print('he')
attention_encodings = [model.encode(sentence) for sentence in attention_array]

print(aita_encodings)
aita_df = pd.DataFrame(aita_encodings)
attention_df = pd.DataFrame(attention_encodings)

# aita_df=aita_df.T
# attention_df = attention_df.T

aita_df.to_csv('aita_encodings.csv', index=False, header=False)
attention_df.to_csv('attention_encodings.csv', index=False, header=False)