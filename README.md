# sentence-classifications
A project examining different methods of sentence classifications. Currently I've implemented an artificial neural network, as well as Euclidian distance.

## Requirements
In order to use this on your own, you will need the following:
Vue3
Flask
numpy
pandas
scikit-learn
PyTorch

## Usage
To spin up the server, navigate to the api directory in the backend and run "flask run", and for the frontend navigate to the frontend directory and run "npm run serve".

## Methods
The data is parsed into sentences and embedded as a vector using the 'paraphrase-MiniLM-L6-v2' model from HuggingFace. From there, the respective data matrices are split up into testing and training sets. 

### Euclidian Algorithm
The Euclidian Algorithm is relatively straightforward. It takes a single mean pooled vector from the training set and uses that as a centroid for each class. In order to classify a data point, a word embedding is matched to the class with the closest centroid. It performs decently on training data, effectively classifying a word 80% of the time from the testing sets.

### ANN
The neural network approach stacks all the training datasets into one huge dataframe, and contrasts that with a long clas vector that holds the labels. That is trained with 700 epochs in Pytorch, with a requirement for a small learning rate. <code> lr=0.00001</code> This is also effective, but seems to be biased towards larger datasets.

## Data Processing and Pipeline
A .txt file from the data directory will be matched and processed if there is both a name.txt file AND the name is added as a string to data_processing/data_config.py's names list. In order to add more categories, simply append both of those. 

The text file is read from the data directory and is split up into sentences (defined as being seperated by punctuation) in the parse_data.py file. From there, the sentences are randomly put into training and testing splits and stored as both plain txt files, and embedded csv files. The centroid of the mean pooled training data frame is stored in the centorids.csv file. All files are written to the data directory.


## Architecture
### Front End
For the frontend there is a vue project. It's pretty simple, a single page with a textarea that sends the users text to the backend and formats it in a table. 

### Back End
The backend is broken into two main components. The API and the models. 
#### Models
The models, data, and data_processing are for the models and for the most part modular. There are some hacky path bindings that are used for imports to the API, so whilst ugly, are a neccessary. 

#### API
The API is a Flask API that has two files. A main app.py that serves the data to the front end, and a Sentence Library that directly interacts with the model component in order to convert the text from the front-end into classes from the models. 
