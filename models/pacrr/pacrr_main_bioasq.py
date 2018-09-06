import os
import sys
import json
import uuid
import copy
import keras
import shutil
import pickle
import gensim
import random
import argparse
import datetime
import subprocess

import numpy as np
import tensorflow as tf

from os import listdir
from tqdm import tqdm
from random import shuffle
from os.path import isfile, join
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors

from pacrr_model import PACRR
from utils.pacrr_utils import pacrr_train, pacrr_predict, map_term2ind, produce_pos_neg_pairs, produce_reranking_inputs, shuffle_train_pairs
from utils.bioasq_utils import *


parser = argparse.ArgumentParser()
parser.add_argument('-config', dest='config_file')
parser.add_argument('-params', dest='params_file')
parser.add_argument('-log', dest='log_name', default='run')
args = parser.parse_args()

config_file = args.config_file
params_file = args.params_file

with open(config_file, 'r') as f:
	config = json.load(f)  
with open(params_file, 'r') as f:
	model_params = json.load(f)  

topk = 100

data_directory = '../../bioasq_data'

bm25_data_path_train = data_directory + '/bioasq_bm25_top{0}.train.pkl'.format(topk, topk)
docset_path_train = data_directory + '/bioasq_bm25_docset_top{0}.train.pkl'.format(topk, topk)

bm25_data_path_dev = data_directory + '/bioasq_bm25_top{0}.dev.pkl'.format(topk, topk)
docset_path_dev = data_directory + '/bioasq_bm25_docset_top{0}.dev.pkl'.format(topk, topk)

bm25_data_path_test = data_directory + '/bioasq_bm25_top{0}.test.pkl'.format(topk, topk)
docset_path_test = data_directory + '/bioasq_bm25_docset_top{0}.test.pkl'.format(topk, topk)

w2v_path = config['WORD_EMBEDDINGS_FILE']
idf_path = config['IDF_FILE']


with open(bm25_data_path_train, 'rb') as f:
	data_train = pickle.load(f)

with open(docset_path_train, 'rb') as f:
	docset_train = pickle.load(f)

with open(bm25_data_path_dev, 'rb') as f:
	data_dev = pickle.load(f)

with open(docset_path_dev, 'rb') as f:
	docset_dev = pickle.load(f)

with open(bm25_data_path_test, 'rb') as f:
	data_test = pickle.load(f)

with open(docset_path_test, 'rb') as f:
	docset_test = pickle.load(f)

with open(idf_path, 'rb') as f:
	idf = pickle.load(f)

print('All data loaded. Pairs generation started..')

# map each term to an id
term2ind = map_term2ind(w2v_path)

# Produce Pos/Neg pairs for the training subset of queries.
print('Producing Pos-Neg pairs for training data..')
train_pairs = produce_pos_neg_pairs(data_train, docset_train, idf, term2ind, model_params, q_preproc_bioasq, d_preproc_bioasq)

# Produce Pos/Neg pairs for the development subset of queries.
print('Producing Pos-Neg pairs for dev data..')
dev_pairs = produce_pos_neg_pairs(data_dev, docset_dev, idf, term2ind, model_params, q_preproc_bioasq, d_preproc_bioasq)

# Produce Pos/Neg pairs for the test subset of queries.
print('Producing Pos-Neg pairs for test data..')
test_pairs = produce_pos_neg_pairs(data_test, docset_test, idf, term2ind, model_params, q_preproc_bioasq, d_preproc_bioasq)

# Produce reranking inputs for the development subset of queries.
print('Producing reranking data for dev..')
dev_reranking_data = produce_reranking_inputs(data_dev, docset_dev, idf, term2ind, model_params, q_preproc_bioasq, d_preproc_bioasq)

# Produce reranking inputs for the test subset of queries.
print('Producing reranking data for test..')
test_reranking_data = produce_reranking_inputs(data_test, docset_test, idf, term2ind, model_params, q_preproc_bioasq, d_preproc_bioasq)

#Random shuffle training pairs
train_pairs = shuffle_train_pairs(train_pairs)

retr_dir = os.path.join('logs', args.log_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
print(retr_dir)
os.makedirs(os.path.join(os.getcwd(), retr_dir))

json_model_params = copy.deepcopy(model_params)
json_model_params['embed'] = []
with open(retr_dir+ '/{0}'.format(params_file.split('/')[-1]), 'w') as f:
	json.dump(json_model_params, f, indent=4)

metrics = ['map', 'P_10', 'P_20', 'ndcg_cut_10', 'ndcg_cut_20']

res = pacrr_train(train_pairs, dev_pairs, dev_reranking_data, term2ind, config, model_params, metrics, retr_dir, doc_id_prefix='http://www.ncbi.nlm.nih.gov/pubmed/')
pacrr_predict(*res, test_reranking_data, term2ind, config, model_params, metrics, retr_dir, doc_id_prefix='http://www.ncbi.nlm.nih.gov/pubmed/')
