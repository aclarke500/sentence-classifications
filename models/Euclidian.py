import pandas as pd
import math
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
import os # adding filepaths list from data_processing directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data_processing import data_config
filepaths = data_config.filepaths
names = data_config.names

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# building dict where keys are the classes, and the values are test dataframes
labels = {}
for name in names:
  labels[name] = pd.read_csv('../data/'+name+'_test_embeddings.csv')


centroids = pd.read_csv('../data/centroids.csv')
# cs_df = pd.read_csv('../data/cs_test_embeddings.csv')
# aita_df = pd.read_csv('../data/aita_test_embeddings.csv')
# attention_df = pd.read_csv('../data/attention_test_embeddings.csv')

# while True:
#   x = 5

def get_prediction_class(word_vec):
  best_distance = math.inf
  classification = None
  for label, centroid in centroids.items():
    distance = np.linalg.norm(np.array(word_vec)-np.array(centroid.tolist()))
    if (distance) < best_distance:
      best_distance = distance
      classification = label
  return classification

for group in labels.keys():
  correct = 0
  wrong = 0
  for index, row in labels[group].iterrows():
    prediction = get_prediction_class(row.tolist())
    if prediction == group:
      correct+=1
    else:
      wrong+=1
  print(f'Score for {group}')
  print(f'Correct:{correct} \t Wrong:{wrong} \t Score:{round(correct/(correct+wrong),2)}')

while True:
    s = input("Enter a word:\t")
    vec = model.encode(s)
    result = get_prediction_class(vec.tolist())
    print(f'Your word class is {result}')