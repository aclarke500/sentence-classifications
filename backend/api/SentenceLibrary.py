import re
import torch
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')



import sys
import os # adding filepaths list from data_processing directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import Euclidian, LoadANN


def embed_sentences(text):
  sentences = split_into_sentences(text)
  vectors = [(model.encode(sentence)) for sentence in sentences]
  return vectors

def get_euclidian_predictions(text):
  embeddings = embed_sentences(text)
  # euclidian expects regular list
  classifications = [Euclidian.get_prediction_class(vec.tolist()) for vec in embeddings]
  return classifications

def get_model_predictions(text):
   embeddings = embed_sentences(text)
   euclidian_classifications = [Euclidian.get_prediction_class(vec.tolist()) for vec in embeddings]
   sentences = split_into_sentences(text)
   ann_classifications = [LoadANN.predict(sentence) for sentence in sentences]

   return {
      "euclidian": euclidian_classifications,
      "ann": ann_classifications,
      "sentences":sentences
   }


def split_into_sentences(text):
    # Define the pattern for splitting: period, exclamation point, or question mark followed by a space or end of string
    sentence_endings = r'[.!?](?:\s|$)'
    sentences = re.split(sentence_endings, text)
    # The re.split method includes an empty string at the end if the text ends with a delimiter; remove it if present
    return [sentence for sentence in sentences if sentence]

