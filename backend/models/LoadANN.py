import torch
from sentence_transformers import SentenceTransformer


import sys
import os # adding filepaths list from data_processing directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import model_definitions
from data_processing import data_config

# from model_definitions import ANNModel  # Adjust the import path if necessary
ANNModel = model_definitions.ANNModel
filepaths = data_config.filepaths
names = data_config.names

# Instantiate the transformer and the model
trans = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model = ANNModel()

# Load the saved state dictionary and set the model to evaluation mode
# /Users/adamclarke/Desktop/Data/sentence-classifications/backend/api
model.load_state_dict(torch.load('/Users/adamclarke/Desktop/Data/sentence-classifications/backend/models/model_state_dict.pth'))
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

def predict(word):
    tensor_embedding = torch.tensor(trans.encode(word))
    with torch.no_grad():
        output = model(tensor_embedding)
        output_list = output.tolist()
        return get_expected_class(output_list)
