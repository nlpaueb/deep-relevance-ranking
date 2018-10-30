import dynet as dy
import pickle
import heapq
import utils
import numpy
import random
import posit as model_
import sys

dataloc = '../../bioasq_data/'

print('Loading Data')
exp_name = 'posit'

with open(dataloc + 'bioasq_bm25_top100.' + sys.argv[2] + '.pkl', 'rb') as f:
  data = pickle.load(f)
with open(dataloc + 'bioasq_bm25_docset_top100.' + sys.argv[2] + '.pkl',
          'rb') as f:
  docs = pickle.load(f)

# Test data is still from year 5, which does not include 2017/2018 abstracts.
utils.RemoveBadYears(data, docs, False)

words = {}
utils.GetWords(data, docs, words)

model = model_.Model(dataloc, words)
model.Load(sys.argv[1])

model.SetDropout(-1.0)
print('Making preds')
json_preds = {}
json_preds['questions'] = []
for i in range(len(data['queries'])):
  dy.renew_cg()

  qtext = data['queries'][i]['query_text']
  qwds, qvecs, qconv = model.MakeInputs(qtext)

  rel_scores = {}
  rel_scores_sum = {}
  for j in range(len(data['queries'][i]['retrieved_documents'])):
    doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
    dtext = (docs[doc_id]['title'] + ' <title> ' +
             docs[doc_id]['abstractText'])
    dwds, dvecs, dconv = model.MakeInputs(dtext)
    bm25 = data['queries'][i]['retrieved_documents'][j]['norm_bm25_score']
    efeats = model.GetExtraFeatures(qtext, dtext, bm25)
    efeats_fv = dy.inputVector(efeats)
    score = model.GetQDScore(qwds, qvecs, qconv, dwds, dvecs, dconv,
                             efeats_fv)
    rel_scores[j] = score.value()

  top = heapq.nlargest(100, rel_scores, key=rel_scores.get)
  utils.JsonPredsAppend(json_preds, data, i, top)
  dy.renew_cg()

utils.DumpJson(json_preds, exp_name + "_test_preds.json")
print('Done')
