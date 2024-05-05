
from gensim.models import Word2Vec
import gensim


warnings.filterwarnings(action='ignore')


# need file to read (dataset)
# Replaces escape character with space
# iterate through each sentence in the file

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count=1,
								vector_size=100, window=5)

# Print results
print("Cosine similarity between 'alice' " +
	"and 'wonderland' - CBOW : ",
	model1.wv.similarity('alice', 'wonderland'))

print("Cosine similarity between 'alice' " +
	"and 'machines' - CBOW : ",
	model1.wv.similarity('alice', 'machines'))

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=100,
								window=5, sg=1)

# Print results
print("Cosine similarity between 'alice' " +
	"and 'wonderland' - Skip Gram : ",
	model2.wv.similarity('alice', 'wonderland'))

print("Cosine similarity between 'alice' " +
	"and 'machines' - Skip Gram : ",
	model2.wv.similarity('alice', 'machines'))
