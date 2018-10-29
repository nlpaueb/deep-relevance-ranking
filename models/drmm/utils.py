import json
import re
import numpy
import dynet as dy
import random

def leaky_relu(x):
  return dy.bmax(.1*x, x)

def uwords(words):
  uw = {}
  for w in words:
    uw[w] = 1
  return [w for w in uw]

def ubigrams(words):
  uw = {}
  prevw = "<pw>"
  for w in words:
    uw[prevw + '_' + w] = 1
    prevw = w
  return [w for w in uw]

# For BioASQ only when using years 1-4 as training and 5 as dev/test.
# Get rid of 2017/2018 plus 2016 in training.
# For competition, training get rid of 2017/2018 and test 2018.
def RemoveBadYears(data, doc_text, train):
  for i in range(len(data['queries'])):
    j = 0
    while True:
      doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
      year = doc_text[doc_id]['publicationDate'].split('-')[0]
      ##########################
      # Skip 2017/2018 docs always. Skip 2016 docs for training.
      # Need to change for final model - 2017 should be a train year only.
      # Use only for testing.
      if year == '2017' or year == '2018' or (train and year == '2016'):
        del data['queries'][i]['retrieved_documents'][j]
      else:
        j += 1
      ##########################
      if j == len(data['queries'][i]['retrieved_documents']):
        break

def GetWords(data, doc_text, words):
  for i in range(len(data['queries'])):
    qwds = tokenize(data['queries'][i]['query_text'])
    for w in qwds:
      words[w] = 1
    for j in range(len(data['queries'][i]['retrieved_documents'])):
      doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
      dtext = (doc_text[doc_id]['title'] + ' <title> ' +
               doc_text[doc_id]['abstractText'])
      dwds = tokenize(dtext)
      for w in dwds:
        words[w] = 1

def RemoveTrainLargeYears(data, doc_text):
  for i in range(len(data['queries'])):
    hyear = 1900
    for j in range(len(data['queries'][i]['retrieved_documents'])):
      if data['queries'][i]['retrieved_documents'][j]['is_relevant']:
        doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
        year = doc_text[doc_id]['publicationDate'].split('-')[0]
        if year[:1] == '1' or year[:1] == '2':
          if int(year) > hyear:
            hyear = int(year)
    j = 0
    while True:
      doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
      year = doc_text[doc_id]['publicationDate'].split('-')[0]
      if (year[:1] == '1' or year[:1] == '2') and int(year) > hyear:
        del data['queries'][i]['retrieved_documents'][j]
      else:
        j += 1
      if j == len(data['queries'][i]['retrieved_documents']):
        break

def GetTrainData(data, max_neg=1):
  train_data = []
  for i in range(len(data['queries'])):
    pos = []
    neg = []
    for j in range(len(data['queries'][i]['retrieved_documents'])):
      is_rel = data['queries'][i]['retrieved_documents'][j]['is_relevant']
      if is_rel:
        pos.append(j)
      else:
        neg.append(j)
    if len(pos) > 0 and len(neg) > 0:
      for p in pos:
        neg_ex = []
        if len(neg) <= max_neg:
          neg_ex = neg
        else:
          used = {}
          while len(neg_ex) < max_neg:
            n = random.randint(0, len(neg)-1)
            if n not in used:
              neg_ex.append(neg[n])
              used[n] = 1
        inst = [i, [p] + neg_ex]
        train_data.append(inst)
  return train_data

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

def tokenize(x):
  return bioclean(x)

def DumpJson(data, fname):
  with open(fname, 'w') as fw:
    json.dump(data, fw, indent=4)

def JsonPredsAppend(preds, data, i, top):
  pref = "http://www.ncbi.nlm.nih.gov/pubmed/"
  qid = data['queries'][i]['query_id']
  query = data['queries'][i]['query_text']
  qdict = {}
  qdict['body'] = query
  qdict['id'] = qid
  doc_list = []
  for j in top:
    doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
    doc_list.append(pref + doc_id)
  qdict['documents'] = doc_list
  preds['questions'].append(qdict)
