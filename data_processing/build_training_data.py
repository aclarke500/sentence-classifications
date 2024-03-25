import pandas as pd
from data_config import names

X_train = pd.DataFrame()
y_train = []
X_test = pd.DataFrame()
y_test = []
idx = 0
for name in names:
  train_df = pd.read_csv('../data/'+name+'_train_embeddings.csv', header=None)
  print(train_df.shape)
  train_y = [idx] * train_df.shape[0]
  y_train+=train_y
  X_train = pd.concat([X_train, train_df], ignore_index=True, axis=0)

  test_df = pd.read_csv('../data/'+name+'_test_embeddings.csv', header=None)
  test_y = [idx] * test_df.shape[0]
  y_test += test_y
  X_test = pd.concat([X_test, test_df], ignore_index=True, axis=0)

  idx+=1

print(X_train.shape)
X_train.to_csv('../data/X_train.csv', index=False)
X_test.to_csv('../data/X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('../data/y_test.csv', index=False)
pd.DataFrame(y_train).to_csv('../data/y_train.csv', index=False)


