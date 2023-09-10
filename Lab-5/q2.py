import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# Specify the filenames of the text files
filename1 = 'doc1.txt'
filename2 = 'doc2.txt'

# Read the contents of the text files
with open(filename1, 'r') as file:
    doc1 = file.read()

with open(filename2, 'r') as file:
    doc2 = file.read()

# Preprocess and tokenize the text
def preprocess(text):
    # Tokenize the text (split into words)
    words = text.split()
    # Remove punctuation and convert to lowercase
    words = [word.lower().strip('.,!?()[]{}"\'') for word in words]
    # Remove stopwords (you may need to define a list of stopwords)
    stopwords = set(["a", "an", "the", "is", "and", "of", "this", "on"])
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

# Preprocess the documents
doc1 = preprocess(doc1)
doc2 = preprocess(doc2)

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

# Calculate cosine distance
cosine_distance = cosine_distances(tfidf_matrix[0], tfidf_matrix[1])

# Cosine similarity is 1 - cosine distance
cosine_similarity = 1 - cosine_distance

print(f"Cosine Distance: {cosine_distance[0][0]:.4f}")