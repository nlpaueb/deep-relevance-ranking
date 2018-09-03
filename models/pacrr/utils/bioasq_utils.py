import re
import sys
import json
import pickle
import random
import operator

from tqdm import tqdm
from gensim.models import KeyedVectors


random.seed(1234)

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

UNK_TOKEN = '*UNK*'

def set_unk_tokens(tokens, term2ind):
	clear_tokens = []
	for t in tokens:
		if t in term2ind:
			clear_tokens.append(t)
		else:
			clear_tokens.append(UNK_TOKEN)
	return clear_tokens

def q_preproc_bioasq(q_text, max_q_len, idf_dict, max_idf, term2ind):
	q_text = bioclean(q_text)
	q_text_unk = set_unk_tokens(q_text, term2ind)
	
	if len(q_text_unk) > max_q_len:
		tok_idf = []
		for token in q_text_unk:
			if token in idf_dict:
				tok_idf.append((token, idf_dict[token]))
			else:
				tok_idf.append((token, max_idf))
		tok_idf.sort(key=lambda tup: tup[1])
		while len(q_text_unk) > max_q_len:
			q_text_unk.remove(tok_idf[0][0])
			tok_idf.pop(0)
	
	q_text_inds = [term2ind[t] for t in q_text_unk][:max_q_len]
	return q_text_inds, q_text

def d_preproc_bioasq(doc_dict, max_doc_len, term2ind):
	d_text = bioclean(doc_dict['title'] + ' ' + doc_dict['abstractText'])
	d_text_unk = set_unk_tokens(d_text, term2ind)
	d_text_inds = [term2ind[t] for t in d_text_unk][:max_doc_len]
	return d_text_inds, d_text

