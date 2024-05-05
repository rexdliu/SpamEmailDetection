from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

def trainword2vec(sentences):
        # new word2vec_model
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
        return model
#save and load the model
        def save_model(model, filename):
            model.save(filename)

        def load_model(filename):
            model = Word2Vec.load(filename)
            return model


