from gensim.models import KeyedVectors
from konlpy.tag import Mecab
from W2VData import tokenize

ADD_POS = True
LOWER = True

SG = 1
MIN_COUNT = 3
EMBED_SIZE = 100
VOCAB_SIZE = 3700

model = KeyedVectors.load("./model/word2vec_sg%s_mc%s_es%s_vc%s_pos_lower.kv" % (SG, MIN_COUNT, EMBED_SIZE, VOCAB_SIZE))

print("vocab size: ", len(model.vocab))
print("vector shape: ", model.vectors.shape)
print()

tokenizer = Mecab()
print("similarity: ", *tokenize(tokenizer, '배우 여배우', ADD_POS, LOWER))
print(model.similarity(*tokenize(tokenizer, '배우 여배우', ADD_POS, LOWER)))
print()

print("most similar: ", *tokenize(tokenizer, '배우', ADD_POS, LOWER))
print(model.most_similar(positive=tokenize(tokenizer, '배우', ADD_POS, LOWER), topn=3))
print()

print("most similar")
print(model.most_similar(positive=tokenize(tokenizer, '배우	남자', ADD_POS, LOWER), negative=tokenize(tokenizer, '여배우', ADD_POS, LOWER), topn=3))
print("embedding\n", model['연기/nng'])

