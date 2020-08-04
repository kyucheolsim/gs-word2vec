# written by kylesim

import re
from gensim.test.utils import datapath

def clean_string(string):
	string = re.sub(r"[^가-힣A-Za-z0-9().,!?+\-\'\"]", " ", string)
	return string

# tokenizer: with member function pos
def tokenize(tokenizer, line, add_pos, lower):
	if add_pos:
		line = ["/".join(tkn) for tkn in tokenizer.pos(line)]
	else:
		line = [tkn[0] for tkn in tokenizer.pos(line)]
	if lower:
		line = [tkn.lower() for tkn in line]
	return line

class W2VData(object):
	def __init__(self, input_path, tokenizer, tokenize, data_idx = 1, add_pos=True, lower=True):
		super(W2VData, self).__init__()
		self.data_path = datapath(input_path)
		self.data_idx = data_idx
		self.tokenizer = tokenizer
		self.tokenize = tokenize
		self.add_pos = add_pos
		self.lower = lower

	def __iter__(self):
		for line in open(self.data_path):
			line = line.split('\t')[self.data_idx]
			line = self.tokenize(self.tokenizer, line, self.add_pos, self.lower)
			yield line

