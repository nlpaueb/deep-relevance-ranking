import dynet as dy
import pickle
import heapq
import utils
import numpy
import random
import posit as model_

dataloc = '../../bioasq_data/'

print('Loading Data')
exp_name = 'posit'

with open(dataloc + 'bioasq_bm25_top100.dev.pkl', 'rb') as f:
  data = pickle.load(f)
with open(dataloc + 'bioasq_bm25_docset_top100.dev.pkl',
          'rb') as f:
  docs = pickle.load(f)
with open(dataloc + 'bioasq_bm25_top100.train.pkl', 'rb') as f:
  tr_data = pickle.load(f)
with open(dataloc + 'bioasq_bm25_docset_top100.train.pkl',
          'rb') as f:
  tr_docs = pickle.load(f)

utils.RemoveBadYears(tr_data, tr_docs, True)
utils.RemoveTrainLargeYears(tr_data, tr_docs)
utils.RemoveBadYears(data, docs, False)

words = {}
utils.GetWords(tr_data, tr_docs, words)
utils.GetWords(data, docs, words)

model = model_.Model(dataloc, words)

################
# TRAIN MODEL
################
print('Training the model')

updates = 0
for epoch in range(40):
  model.SetDropout(0.3)

  num_docs = 0
  relevant = 0
  returned = 0
  brelevant = 0
  breturned = 0
  train_examples = utils.GetTrainData(tr_data, 1)
  random.shuffle(train_examples)
  loss = []
  for ex in train_examples:
    i = ex[0]
    qtext = tr_data['queries'][i]['query_text']
    qwds, qvecs, qconv = model.MakeInputs(qtext, True)

    pos = []
    neg = []
    best_neg = -1000000.0
    for j in ex[1]:
      is_rel = tr_data['queries'][i]['retrieved_documents'][j]['is_relevant']
      doc_id = tr_data['queries'][i]['retrieved_documents'][j]['doc_id']
      dtext = (tr_docs[doc_id]['title'] + ' <title> ' +
               tr_docs[doc_id]['abstractText'])
      dwds, dvecs, dconv = model.MakeInputs(dtext, True)
      bm25 = (tr_data['queries'][i]['retrieved_documents'][j]
              ['norm_bm25_score'])
      efeats = model.GetExtraFeatures(qtext, dtext, bm25)
      efeats_vec = dy.inputVector(efeats)
      score = model.GetQDScore(qwds, qvecs, qconv, dwds, dvecs, dconv,
                               efeats_vec, True)
      if is_rel:
        pos.append(score)
      else:
        neg.append(score)
        if score.value() > best_neg:
          best_neg = score.value()

    if pos[0].value() > best_neg:
      relevant += 1
      brelevant += 1
    returned += 1
    breturned += 1

    num_docs += 1

    if len(pos) > 0 and len(neg) > 0:
      model.PairAppendToLoss(pos, neg, loss)

    if num_docs % 64 == 0 or num_docs == len(train_examples):
      model.UpdateBatch(loss)
      updates += 1
      loss = []

    if num_docs % 64 == 0:
      print('Epoch %d' % epoch +
            ', Instances %d' % num_docs +
            ', Cumulative Acc %f' % (float(relevant)/float(returned)) +
            ', Sub-epoch Acc %f' % (float(brelevant)/float(breturned)))
      brelevant = 0
      breturned = 0

  print('End of epoch %d' % epoch +
        ', Total train docs %d' % num_docs +
        ', Train Acc %f' % (float(relevant)/float(returned)))

  model.SetDropout(-1.0)
  print('Saving model')
  model.Save("model_" + exp_name + "_ep" + str(epoch))
  print('Model saved')

  ################
  # MAKE DEV PREDS
  ################
  model.SetDropout(-1.0)
  print('Making Dev preds')
  json_preds = {}
  json_preds['questions'] = []
  num_docs = 0
  for i in range(len(data['queries'])):
    num_docs += 1
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
      dy.renew_cg()

    top = heapq.nlargest(100, rel_scores, key=rel_scores.get)
    utils.JsonPredsAppend(json_preds, data, i, top)
    dy.renew_cg()

  utils.DumpJson(json_preds, exp_name + "_" + str(epoch) + '.json')
  print('Done')
