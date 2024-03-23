# sentence-classifications
A project using a sentence transformer to classify different sentences based on their embeddings. The current implementation classifies sentences as belonging to either the "Am I the Asshole" subreddit OR the original 'Attention is All You Need' transformer paper.

## Methods
The data is parsed into sentences and embedded as a vector using <code> model = SentenceTransformer('paraphrase-MiniLM-L6-v2') </code>. From there, the respective data matrices are split up into testing and training sets, and the training datasets are mean pooled. This mean pooled vector acts as a centroid for each source. From there, these centroids are used to compare each testing sentence using Euclidian distance and each sentence is classified as whichever centroid it is closest to.
## Data Pipeline
The data is gathered as raw text and thrown into a txt file. Then, parse_text.py takes these text files, and finds individual sentences, and encodes the sentences. It then writes these encoidings to a csv file.
