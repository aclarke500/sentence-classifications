# sentence-classifications
A project using a sentence transformer to classify different sentences based on their embeddings. The current implementation classifies sentences as belonging to either the "Am I the Asshole" subreddit OR the original 'Attention is All You Need' transformer paper.

## Methods
The data is parsed into sentences and embedded as a vector using <code> model = SentenceTransformer('paraphrase-MiniLM-L6-v2') </code>. From there, the respective data matrices are split up into testing and training sets, and the training datasets are mean pooled. This mean pooled vector acts as a centroid for each source. From there, these centroids are used to compare each testing sentence using Euclidian distance and each sentence is classified as whichever centroid it is closest to.
## Data Processing and Pipeline
A .txt file from the data directory will be matched and processed if there is both a name.txt file AND the name is added as a string to data_processing/data_config.py's names list. In order to add more categories, simply append both of those. 

The text file is read from the data directory and is split up into sentences (defined as being seperated by punctuation) in the parse_data.py file. From there, the sentences are randomly put into training and testing splits and stored as both plain txt files, and embedded csv files. The centroid of the mean pooled training data frame is stored in the centorids.csv file. All files are written to the data directory.
