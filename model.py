import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the pre-trained Word2Vec model
model = Word2Vec.load("word2vec_email.model")

# convert emails into vectors
def email_to_vector(email, model):
    words = [word for word in email if word in model.wv]
    if words:
        return np.mean(model.wv[words], axis=0)
    else:
        return np.zeros(model.vector_size)

# Load dataset based on vocabulary
data = pd.read_csv('')

# Assume 'Message' is the column with emails and 'categories' is the column with labels
data['processed'] = data['Message'].apply(lambda x: x.lower().split())  # Simple tokenization and case normalization

# Convert each email to a vector
data['vector'] = data['processed'].apply(lambda x: email_to_vector(x, model))

# Prepare data for the MLP
X = np.vstack(data['vector'])
y = data['categories'].map({'spam': 1, 'ham': 0}).values  # Convert labels to numeric

#  Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Build the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=500, random_state=1)

# Train the MLP classifier
mlp.fit(X_train, y_train)

# Make predictions
predictions = mlp.predict(X_test)
predictions_probablity=mlp.predict_proba(X_test)[:,1]#probablity of spam
# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, target_names=['ham', 'spam'])

print("Accuracy:", accuracy)
print("Classification Report:\n", report)



