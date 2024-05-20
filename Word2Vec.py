import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Load the dataset
try:
    data = pd.read_csv('spam.csv')
    print("Columns in the dataset:", data.columns)  # Verifying the column names
except FileNotFoundError:
    print("The specified file was not found.")
    exit()

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

# Checking the preprocessed data
print(data['processed'].head())

# Convert 'categories' to numeric labels
data['label'] = data['Category'].map({'spam': 1, 'ham': 0})

# Check if there's any issue with label conversion
if data['label'].isnull().any():
    print("There are unhandled categories in 'Category' column.")

# Train the Word2Vec model
vector_size = 300  # Using a more typical vector size for Word2Vec
model = Word2Vec(sentences=data['processed'], vector_size=vector_size, window=5, min_count=2, workers=4, sg=0)  # CBOW model
model.save("word2vec_email.model")
print("Word2Vec model trained and saved.")
