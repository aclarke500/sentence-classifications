import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.spatial.distance import euclidean

aita_df = pd.read_csv('aita_encodings.csv', header=None)
attention_df = pd.read_csv('attention_encodings.csv', header=None)


aita_train_df, aita_test_df = train_test_split(aita_df, test_size=0.25, random_state=42)
attention_train_df, attention_test_df = train_test_split(attention_df, test_size=0.25, random_state=42)

aita_vec = aita_train_df.mean(axis=0)
attention_vec = attention_train_df.mean(axis=0)

# aita is positve
tp = 0
tn = 0
fp = 0
fn = 0

for i, row in aita_test_df.iterrows():
  aita_dist = euclidean(row, aita_vec)
  attention_dist = euclidean(row, attention_vec)
  print(f'p: {aita_dist}, \t n:{attention_dist}')
  if aita_dist < attention_dist:
    tp+=1
    print('fn')
  else:
    fn +=1
    print('tp')
    
  
print('attention')
for i, row in attention_test_df.iterrows():
  aita_dist = euclidean(row, aita_vec)
  attention_dist = euclidean(row, attention_vec)

  print(f'p: {aita_dist}, \t n:{attention_dist}')
  # what we want
  if aita_dist > attention_dist: 
    tn+=1
    print('fp')
  else:
    
    print('tn')
    fp +=1

print(f'True Positive: {tp} \t False Positive: {fp}')
print(f'False Negative:{fn} \t True Negative:{tn}')

