import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from MLP import MLP

# Load the dataset
data = pd.read_csv('spam.csv')

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

data['processed'] = data['Message'].apply(preprocess)
data['label'] = data['Category'].map({'spam': 1, 'ham': 0})

# Load the pre-trained Word2Vec model
model = Word2Vec.load("word2vec_email.model")

def email_to_vector(email):
    words = preprocess(email)
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

X = np.array([email_to_vector(email) for email in data['Message']])
y = data['label'].values.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the MLP
input_size = X_train.shape[1]
hidden_size = 50
output_size = 1

Model = MLP(input_size, hidden_size, output_size)
Model.train(X_train, y_train, epochs=10000, learning_rate=0.01)

# Evaluate the model
predictions = Model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, target_names=['ham', 'spam'])
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
