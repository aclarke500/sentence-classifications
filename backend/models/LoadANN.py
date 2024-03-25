import torch
from sentence_transformers import SentenceTransformer
from model_definitions import ANNModel  # Adjust the import path if necessary

import sys
import os # adding filepaths list from data_processing directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data_processing import data_config
filepaths = data_config.filepaths
names = data_config.names

# Instantiate the transformer and the model
trans = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model = ANNModel()

# Load the saved state dictionary and set the model to evaluation mode
model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval()


sentences = [
  "Hello, World!",
  "I want to make money at my internship at FAANG",
  "Gender and sexuality",
]

# Encode sentences and convert them to a tensor
vectors = torch.stack([torch.tensor(trans.encode(sentence)) for sentence in sentences])

# Check if CUDA is available and move the tensors and model to GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vectors = vectors.to(device)
model.to(device)


def get_expected_class(prediction):
    classification = names[prediction.index(max(prediction))]
    return(classification)



# Make predictions
with torch.no_grad():
    output = model(vectors)
    preds = output.tolist()
    for pred in preds:
        print(get_expected_class(pred))

    while True:
        sentence = input("Enter a sentence:\t")
        vector = trans.encode(sentence)
        output = model(torch.tensor(vector))
        print(output)
        print(get_expected_class(output.tolist()))
