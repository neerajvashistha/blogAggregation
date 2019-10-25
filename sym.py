# spacy
import spacy
nlp = spacy.load('en_core_web_lg')

def most_similar(word):
	queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
	by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
	return by_similarity[:10]


mostSim = [w.lower_ for w in most_similar(nlp.vocab[u'dog'])]

# Gensim

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleGoogleNews-vectors-negative300.bin', binary=True)

#vector = model['easy']
model.similarity('aWord','aWord2')
model.most_similar('dog')

# NLTK

from nltk.corpus import wordnet as wn
print wn.synset("eat.v.01").lemma_names # prints synonyms of eat

# PyDictionary

from PyDictionary import PyDictionary

dictionary=PyDictionary()
dictionary.synonym("Life")
