from gensim.models import Word2Vec

# Load the saved Word2Vec model
word2vec_model = Word2Vec.load("word2vec_code.model")

# Test: Find similar words
print(word2vec_model.wv.most_similar("throw"))  # Example: Words similar to 'ssl'

# Get vector representation of a word
vector = word2vec_model.wv["return"]
print(vector)