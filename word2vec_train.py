from konlpy.tag import Mecab
from gensim.models.word2vec import Word2Vec
from W2VData import tokenize, W2VData
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ADD_POS = True
LOWER = True

SG = 0
ITER = 10
MIN_COUNT = 3
EMBED_SIZE = 100
VOCAB_SIZE = 10000

tokenizer = Mecab()
sentences = W2VData('./data/nsmc/ratings_all.txt', tokenizer, tokenize, ADD_POS, LOWER)

# train
model = Word2Vec(sentences = sentences, size = EMBED_SIZE, window = 5, min_count = MIN_COUNT, max_vocab_size = VOCAB_SIZE, max_final_vocab = VOCAB_SIZE, workers = 4, sg = SG, iter = ITER)

# save lots of memory (not trainable)
model.init_sims(replace = True)

# save KeyedVectors (small and fast load, not trainable) instead of full model
model.wv.save("./model/word2vec_sg%s_mc%s_es%s_vc%s_pos_lower.kv" % (SG, MIN_COUNT, EMBED_SIZE, len(model.wv.vocab)))

