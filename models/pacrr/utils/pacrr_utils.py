import os
import gc
import sys
import json
import copy
import keras
import pickle
import gensim
import random
import operator
import argparse
import datetime
import subprocess

import numpy as np
import tensorflow as tf

from os import listdir
from tqdm import tqdm
from random import shuffle

from os.path import isfile, join
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ReduceLROnPlateau
from keras.backend.tensorflow_backend import set_session
from pacrr_model import PACRR


UNK_TOKEN = '*UNK*'

SEED = 1234
random.seed(SEED)

def map_term2ind(w2v_path):
	word_vectors = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
	vocabulary  = sorted(list(word_vectors.vocab.keys()))

	term2ind = dict([t[::-1] for t in enumerate(vocabulary, start=1)])
	term2ind[UNK_TOKEN] = max(term2ind.items(), key=operator.itemgetter(1))[1] + 1	# Index of *UNK* token

	print('Vocabulary size: {0}'.format(len(vocabulary)))

	return term2ind

def get_idf_list(tokens, idf_dict, max_idf):
	idf_list = []
	for t in tokens:
		if t in idf_dict:
			idf_list.append(idf_dict[t])
		else:
			idf_list.append(max_idf)

	return idf_list

def get_overlap_features(q_tokens, d_tokens, q_idf):
	
	# Map term to idf before set() change the term order
	q_terms_idf = {}
	for i in range(len(q_tokens)):
		q_terms_idf[q_tokens[i]] = q_idf[i]

	# Query Uni and Bi gram sets
	query_uni_set = set()
	query_bi_set = set()
	for i in range(len(q_tokens)-1):
		query_uni_set.add(q_tokens[i])
		query_bi_set.add((q_tokens[i], q_tokens[i+1])) 
	query_uni_set.add(q_tokens[-1])

	# Doc Uni and Bi gram sets
	doc_uni_set = set()
	doc_bi_set = set()
	for i in range(len(d_tokens)-1):
		doc_uni_set.add(d_tokens[i])
		doc_bi_set.add((d_tokens[i], d_tokens[i+1]))
	doc_uni_set.add(d_tokens[-1])

	unigram_overlap = 0
	idf_uni_overlap = 0
	idf_uni_sum = 0
	for ug in query_uni_set:
		if ug in doc_uni_set:
			unigram_overlap += 1
			idf_uni_overlap += q_terms_idf[ug]
		idf_uni_sum += q_terms_idf[ug]
	unigram_overlap /= len(query_uni_set)
	idf_uni_overlap /= idf_uni_sum

	bigram_overlap = 0
	for bg in query_bi_set:
		if bg in doc_bi_set:
			bigram_overlap += 1
	try:
		bigram_overlap /= len(query_bi_set)
	except ZeroDivisionError:
		bigram_overlap = 0

	return [unigram_overlap, bigram_overlap, idf_uni_overlap]

def produce_pos_neg_pairs(data, docset, idf_dict, term2ind, model_params, query_preprocessing, doc_preprocessing):
	
	pairs_list = []

	query_list = []
	query_idf_list = []

	pos_doc_list = []
	neg_doc_list = []

	pos_doc_bm25_list = []
	neg_doc_bm25_list = []

	pos_doc_normBM25_list = []
	neg_doc_normBM25_list = []

	pos_doc_overlap_list = []
	neg_doc_overlap_list = []

	max_idf = max(idf_dict.items(), key=operator.itemgetter(1))[1]

	for q in data['queries']:
		
		rel_ret_set = []
		non_rel_set = []

		rel_set = set(q['relevant_documents'])
		for d in q['retrieved_documents']:
			doc_id = d['doc_id']
			if doc_id in rel_set:
				rel_ret_set.append(d)
			else:
				non_rel_set.append(d)

		query_inds, query_tokens = query_preprocessing(q['query_text'], model_params['maxqlen'], idf_dict, max_idf, term2ind)
		query_idf = get_idf_list(query_tokens, idf_dict, max_idf)

		if not query_inds:
			continue

		not_found_pos = 0
		for pos_doc in rel_ret_set:
			pos_doc_id = pos_doc['doc_id']
			if pos_doc['doc_id'] not in docset:
				not_found_pos += 1
				continue
			if non_rel_set:
				neg_doc = random.choice(non_rel_set)
				neg_doc_id = neg_doc['doc_id']

			pairs_list.append({'pos': pos_doc_id, 'neg': neg_doc_id})

			pos_doc_inds, pos_doc_tokens = doc_preprocessing(docset[pos_doc_id], model_params['simdim'], term2ind)
			neg_doc_inds, neg_doc_tokens = doc_preprocessing(docset[neg_doc_id], model_params['simdim'], term2ind)

			pos_doc_BM25 = pos_doc['bm25_score']
			neg_doc_BM25 = neg_doc['bm25_score']

			pos_doc_normBM25 = pos_doc['norm_bm25_score']
			neg_doc_normBM25 = neg_doc['norm_bm25_score']

			pos_doc_overlap = get_overlap_features(query_tokens, pos_doc_tokens, query_idf)
			neg_doc_overlap = get_overlap_features(query_tokens, neg_doc_tokens, query_idf)

			query_list.append(query_inds)
			query_idf_list.append(query_idf)
			
			pos_doc_list.append(pos_doc_inds)
			pos_doc_bm25_list.append(pos_doc_BM25)
			pos_doc_normBM25_list.append(pos_doc_normBM25)
			pos_doc_overlap_list.append(pos_doc_overlap)

			neg_doc_list.append(neg_doc_inds)
			neg_doc_bm25_list.append(neg_doc_BM25)
			neg_doc_normBM25_list.append(neg_doc_normBM25)
			neg_doc_overlap_list.append(neg_doc_overlap)
		if not_found_pos > 0:
			print('{0} relevant documents are not in the docset.'.format(not_found_pos))

	pairs_data = {
		'queries': query_list,
		'queries_idf': query_idf_list,
		'pos_docs': pos_doc_list,
		'neg_docs': neg_doc_list,
		'pos_docs_BM25': pos_doc_bm25_list,
		'pos_docs_normBM25': pos_doc_normBM25_list,
		'pos_docs_overlap': pos_doc_overlap_list,
		'neg_docs_BM25': neg_doc_bm25_list,
		'neg_docs_normBM25': neg_doc_normBM25_list,
		'neg_docs_overlap': neg_doc_overlap_list,
		'pairs': pairs_list,
		'num_pairs': len(pairs_list)
	}

	return pairs_data

def produce_reranking_inputs(data, docset, idf_dict, term2ind, model_params, query_preprocessing, doc_preprocessing):

	query_data_list = []

	max_idf = max(idf_dict.items(), key=operator.itemgetter(1))[1]

	for q in data['queries']:
		query_data = {}
		
		q_id = q['query_id']
		query_inds, query_tokens = query_preprocessing(q['query_text'], model_params['maxqlen'], idf_dict, max_idf, term2ind)
		query_idf = get_idf_list(query_tokens, idf_dict, max_idf)
		
		doc_id_list = []
		doc_list = []
		doc_BM25_list = []
		doc_norm_BM25_list = []
		doc_overlap_list = []
		for doc in q['retrieved_documents']:
			doc_inds, doc_tokens = doc_preprocessing(docset[doc['doc_id']], model_params['simdim'], term2ind)
			doc_BM25 = doc['bm25_score']
			doc_normBM25 = doc['norm_bm25_score']
			doc_overlap = get_overlap_features(query_tokens, doc_tokens, query_idf)
			
			doc_id_list.append(doc['doc_id'])
			doc_list.append(doc_inds)
			doc_BM25_list.append(doc_BM25)
			doc_norm_BM25_list.append(doc_normBM25)
			doc_overlap_list.append(doc_overlap)
				

		query_data['id'] = q_id
		query_data['token_inds'] = query_inds
		query_data['idf'] = query_idf
		query_data['retrieved_documents'] = {'doc_ids': doc_id_list,
		                                     'doc_list': doc_list,
		                                     'doc_BM25': doc_BM25_list,
		                                     'doc_normBM25': doc_norm_BM25_list,
		                                     'doc_overlap': doc_overlap_list,
		                                     'n_ret_docs': len(doc_id_list)}

		query_data_list.append(query_data)

	return query_data_list

def log(data, retr_dir, filename, cl=False):
	f = open(retr_dir + '/{0}'.format(filename), 'a')
	f.write(data + '\n')
	if cl:
		f.write('\n')
	f.close()

def results_to_string(prefix, metrics, results):
	new_str = '\t'.join(['{0} {1}: {2}'.format(prefix, m, results[m]) for m in metrics])
	return new_str

def myprint(s):
    with open('log_{0}.txt'.format(params_file.split('/')[-1], 'w')) as f:
        print(s, file=f)

def write_trec_eval_results(q_id, sorted_retr_scores, retr_dir, filename):

	path = '{0}/{1}'.format(retr_dir, filename)
	file = open(path, 'a')
	i = 1
	for doc in sorted_retr_scores:
		print("{0} Q0 {1} {2} {3} PACRR".format(q_id, doc[0], i, doc[1]), end="\n", file=file)
		i += 1
	file.close()
	return path

def shuffle_train_pairs(train_data_dict):
	num_pairs = train_data_dict['num_pairs']
	inds = np.arange(num_pairs)     
	np.random.shuffle(inds)              
	for k in train_data_dict.keys():
		if isinstance(train_data_dict[k], list) and len(train_data_dict[k]) == train_data_dict['num_pairs']:
			train_data_dict[k] = np.array(train_data_dict[k])[inds]
	return train_data_dict

def write_bioasq_results_dict(bioasq_res_dict, retr_dir, filename):

	path = '{0}/{1}'.format(retr_dir, filename.replace('dev_', 'dev_bioasq_'))
	with open(path, 'w') as f:
		json.dump(bioasq_res_dict, f, indent=2)
	return path

def trec_eval_custom(q_rels_file, metrics, path):
	eval_res = subprocess.Popen(
		['python', '../../eval/run_eval.py', q_rels_file, path],
		stdout=subprocess.PIPE, shell=False)
	(out, err) = eval_res.communicate()
	eval_res = out.decode("utf-8")

	results = {}

	for line in eval_res.split('\n'):
	
		splitted_line = line.split()
		try:
			first_element = splitted_line[0]
			for metric in metrics:
				if first_element == metric:
					value = float(splitted_line[2])
					results[metric] = value
		except:
			continue		

	file = open(path + '_trec_eval', 'w')
	file.write(eval_res)
	file.close()
	return results

def get_precision_at_k(res_dict, qrels, k):
	custom_metrics = {}
	
	sum_prec_at_k = 0
	for q in res_dict['questions']:
		hits = sum([1 for doc_id in q['documents'][:k] if doc_id in qrels[q['id']]])
		sum_prec_at_k += hits / k
	return sum_prec_at_k / len(res_dict['questions'])

def load_qrels(path):

	with open(path , 'r') as f:
		data = json.load(f)

	qrels = {}
	for q in data['questions']:
		qrels[q['id']] = set(q['documents'])
	return qrels

def load_embeddings(path, term2ind):
	word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
	word2vec.init_sims(replace=True)
	
	# Retrieve dimension space of embedding vectors
	dim = word2vec['common'].shape[0]
	
	# Initialize (with zeros) embedding matrix for all vocabulary
	embedding = np.zeros((len(term2ind)+1, dim))
	rand_count = 0

	# Fill embedding matrix with know embeddings
	for key, value in term2ind.items():
		if value == 0:
			print('ZERO ind found in vocab.')
		try:
			embedding[value] = word2vec[key]
		except:
			rand_count += 1
			continue

	embedding[-1, :] = np.mean(embedding[1:-1, :], axis=0)

	print("No of OOV tokens: %d"%rand_count)
	return embedding

def rerank_query_generator(filename):
	with open(filename, 'rb') as f:
		unpickler = pickle.Unpickler(f)
		i = 0
		while True:
			try:
				q = unpickler.load()
				yield q
				q = None
			except EOFError:
				break	

def rerank(queries_to_rerank, filename, scoring_model, qrels_file, model_params, metrics, retr_dir, doc_id_prefix):
		
		res_dict = {'questions': []}
		for q in tqdm(queries_to_rerank, desc='queries'):
			
			scores = scoring_model.predict(
					{
						'query_inds': pad_sequences(np.tile(q['token_inds'], (len(q['retrieved_documents']['doc_list']), 1)), maxlen=model_params['maxqlen'], padding=model_params['padding_mode']),
						'query_idf': np.expand_dims(pad_sequences(np.tile(q['idf'], (len(q['retrieved_documents']['doc_list']), 1)), model_params['maxqlen'], padding=model_params['padding_mode']), axis=-1),
						'pos_doc_inds': pad_sequences(q['retrieved_documents']['doc_list'], maxlen=model_params['simdim'], padding=model_params['padding_mode']),
						'pos_doc_bm25_scores': np.array(q['retrieved_documents']['doc_normBM25']), 
						'pos_doc_overlap_vec' : np.array(q['retrieved_documents']['doc_overlap'])
					}
					,
					batch_size=model_params['predict_batch_size']

				)
			scores = [s[0] for s in scores]
			retr_scores = list(zip(q['retrieved_documents']['doc_ids'], scores))
			shuffle(retr_scores) # Shuffle docs to make sure re-ranker works.
			sorted_retr_scores = sorted(retr_scores, key=lambda x: x[1], reverse=True)

			res_dict['questions'].append({'id': q['id'], 'documents': [doc_id_prefix + d[0] for d in sorted_retr_scores]})

		path_bioasq = write_bioasq_results_dict(res_dict, retr_dir, filename)
		
		trec_eval_metrics = trec_eval_custom(qrels_file, metrics, path_bioasq)
		
		reported_results = {'map': trec_eval_metrics['map'], 'P_10': trec_eval_metrics['P_10'], 'P_20': trec_eval_metrics['P_20'],'ndcg_cut_10': trec_eval_metrics['ndcg_cut_10'], 'ndcg_cut_20': trec_eval_metrics['ndcg_cut_20']}

		return reported_results

def pacrr_train(train_pairs, dev_pairs, queries_to_rerank, term2ind, config, model_params, metrics, retr_dir, doc_id_prefix=''):

	class Evaluate(keras.callbacks.Callback):
		def on_epoch_end(self, epoch, logs={}):
			epoch += 1
			path = '{0}/{1}'.format(retr_dir, 'weights.keras.epoch_{0}'.format(epoch))
			train_rerank = False

			dev_rerank_results = rerank(queries_to_rerank, 'dev_ranking_epoch{0}'.format(epoch), scoring_model, config['QRELS_DEV'], model_params, metrics, retr_dir, doc_id_prefix)
			dev_res_str = results_to_string('Dev', metrics, dev_rerank_results)

			print('Epoch: {0} \n{1} \nTraining_loss: {2} \t Training_acc: {3} \nVal_loss: {4} \t Val_Acc: {5}'.format(epoch, '\n'.join(dev_res_str.split('\t')), logs['loss'], logs['acc'], logs['val_loss'], logs['val_acc']))
			log('|'.join(list(map(str, [epoch, dev_rerank_results['map'], dev_rerank_results['P_10'], dev_rerank_results['ndcg_cut_10'], dev_rerank_results['P_20'], dev_rerank_results['ndcg_cut_20'], logs['val_acc'], logs['val_loss'], logs['acc'], logs['loss']]))), retr_dir, 'log.txt')
			
			if dev_rerank_results['map'] > self.best_map:
				print('========== Best epoch so far: {0} =========='.format(epoch))
				if self.best_map_weights_path is not None:
					if os.path.exists(self.best_map_weights_path):
						os.remove(self.best_map_weights_path)
				self.best_epoch = epoch
				self.best_map = dev_rerank_results['map']
				self.best_map_weights_path = path
				pacrr_model.dump_weights(path)
				
			print('\n')

	pacrr_train_labels = np.ones((len(train_pairs['queries']), 1), dtype=np.int32)
	pacrr_dev_labels = np.ones((len(dev_pairs['queries']), 1), dtype=np.int32)

	print('Number of samples: {0}'.format(len(train_pairs['queries'])))

	model_params['nsamples'] = train_pairs['num_pairs']
	model_params['embed'] = load_embeddings(config['WORD_EMBEDDINGS_FILE'], term2ind)
	model_params['vocab_size'] = model_params['embed'].shape[0]
	model_params['emb_dim'] = model_params['embed'].shape[1]

	pacrr_model = PACRR(model_params, 3)
	training_model, scoring_model = pacrr_model.build()
	
	training_model.summary()	
	    
	train_params = {}
	print(model_params)
	log(str(model_params), retr_dir, 'model.txt', True)
	log('epoch|dev_map|dev_P_10|dev_ndcg_10|dev_P_20|dev_ndcg_20|dev_acc|dev_loss|train_acc|train_loss', retr_dir, 'log.txt')
	log('epoch|test_map|test_P_10|test_ndcg_10|test_P_20|test_ndcg_20', retr_dir, 'test_log.txt')

	train_params['query_inds'] = pad_sequences(train_pairs['queries'], maxlen=model_params['maxqlen'], padding=model_params['padding_mode'])
	train_params['query_idf'] = np.expand_dims(pad_sequences(train_pairs['queries_idf'], maxlen=model_params['maxqlen'], padding=model_params['padding_mode']), axis=-1)
	train_params['pos_doc_inds'] = pad_sequences(train_pairs['pos_docs'], maxlen=model_params['simdim'], padding=model_params['padding_mode'])
	train_params['pos_doc_bm25_scores'] = np.array(train_pairs['pos_docs_normBM25'])
	train_params['pos_doc_overlap_vec'] = np.array(train_pairs['pos_docs_overlap'])
	for n in range(model_params['numneg']):
		train_params['neg{0}_doc_inds'.format(n)] = pad_sequences(train_pairs['neg_docs'], maxlen=model_params['simdim'], padding=model_params['padding_mode'])
		train_params['neg{0}_doc_bm25_scores'.format(n)] = np.array((train_pairs['neg_docs_normBM25']))
		train_params['neg{0}_doc_overlap_vec'.format(n)] = np.array((train_pairs['neg_docs_overlap']))

	dev_params = {}
	dev_params['query_inds'] = pad_sequences(dev_pairs['queries'], maxlen=model_params['maxqlen'], padding=model_params['padding_mode'])
	dev_params['query_idf'] = np.expand_dims(pad_sequences(dev_pairs['queries_idf'], maxlen=model_params['maxqlen'], padding=model_params['padding_mode']), axis=-1)
	dev_params['pos_doc_inds'] = pad_sequences(dev_pairs['pos_docs'], maxlen=model_params['simdim'], padding=model_params['padding_mode'])
	dev_params['pos_doc_bm25_scores'] = np.array(dev_pairs['pos_docs_normBM25'])
	dev_params['pos_doc_overlap_vec'] = np.array(dev_pairs['pos_docs_overlap'])
	for n in range(model_params['numneg']):
		dev_params['neg{0}_doc_inds'.format(n)] = pad_sequences(dev_pairs['neg_docs'], maxlen=model_params['simdim'], padding=model_params['padding_mode'])
		dev_params['neg{0}_doc_bm25_scores'.format(n)] = np.array((dev_pairs['neg_docs_normBM25']))
		dev_params['neg{0}_doc_overlap_vec'.format(n)] = np.array((dev_pairs['neg_docs_overlap']))

	for k in train_params:
		print(k, train_params[k].shape)

	eval = Evaluate()
	eval.best_epoch = -1
	eval.best_map = 0
	eval.best_map_weights_path = None

	tbcallback = keras.callbacks.TensorBoard(log_dir=retr_dir)

	training_model.fit(train_params, pacrr_train_labels, 
		validation_data=[dev_params, pacrr_dev_labels], 
		batch_size=model_params['train_batch_size'], 
		epochs=model_params['epochs'], 
		callbacks=[eval, tbcallback], 
		verbose=1, 
		shuffle=True)

	return scoring_model, eval.best_map_weights_path, eval.best_epoch
	
def pacrr_predict(scoring_model, weights_path, best_epoch, queries_to_rerank, term2ind, config, model_params, metrics, retr_dir, doc_id_prefix=''):

	scoring_model.load_weights(weights_path)
	rerank_res = rerank(queries_to_rerank, 'test_ranking_epoch{0}'.format(best_epoch), scoring_model, config['QRELS_TEST'], model_params, metrics, retr_dir, doc_id_prefix)
	res_str = results_to_string('Test', metrics, rerank_res)
	print('Test evaluation:')
	print('Best epoch on dev: {0} \n{1} '.format(best_epoch, '\n'.join(res_str.split('\t'))))
	log('|'.join(list(map(str, [best_epoch, rerank_res['map'], rerank_res['P_10'], rerank_res['ndcg_cut_10'], rerank_res['P_20'], rerank_res['ndcg_cut_20']]))), retr_dir, 'test_log.txt')

	return rerank_res