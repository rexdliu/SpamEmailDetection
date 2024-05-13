import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Load the dataset
data = pd.read_csv('spam.csv')
print("Columns in the dataset:", data.columns)  # Verifying the column names

# Preprocess the data
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords and non-alphabetical characters
    return tokens

data['processed'] = data['Message'].apply(preprocess)

# Convert 'categories' to numeric labels
data['label'] = data['Category'].map({'spam': 1, 'ham': 0})

# Train the Word2Vec model
model = Word2Vec(sentences=data['processed'], vector_size=1000, window=5, min_count=2, workers=4)
model.save("word2vec_email.model")
print("Word2Vec model trained and saved.")
