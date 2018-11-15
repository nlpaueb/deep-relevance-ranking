from gensim.models.keyedvectors import KeyedVectors
import pickle
import numpy
import dynet as dy
import utils
import random

class Model:
  def __init__(self, dataloc, words):
    print('Loading word vectors')
    wv = KeyedVectors.load_word2vec_format(
        dataloc + 'pubmed2018_w2v_30D/pubmed2018_w2v_30D.bin',
        binary=True) # C binary format
    self.wv = {}
    for w in words:
      if w in wv:
        self.wv[w] = wv[w]
    wv = None

    print('Loading IDF tables')
    idf = {}
    with open(dataloc + 'IDF.pkl', 'rb') as f:
      idf = pickle.load(f)
    self.idf = {}
    for w in words:
      if w in idf:
        self.idf[w] = idf[w]
    self.max_idf = 0.0
    for w in idf:
      if idf[w] > self.max_idf:
        self.max_idf = idf[w]
    idf = None
    print('Loaded idf tables with max idf %f' % self.max_idf)

    self.model = dy.ParameterCollection()
    self.lr = 0.01
    self.trainer = dy.AdamTrainer(self.model, self.lr)

    self.conv_dim = 30
    self.lstm_dim = 15
    self.W_conv = self.model.add_parameters((self.conv_dim, 3 * self.conv_dim))
    self.b_conv = self.model.add_parameters((self.conv_dim))
    self.pad = self.model.add_lookup_parameters((2, self.conv_dim))

    self.W_gate = self.model.add_parameters((1, self.conv_dim + 1))

    self.biRNN = dy.BiRNNBuilder(2, self.conv_dim, self.conv_dim, self.model,
                                 dy.LSTMBuilder)

    self.W_term1 = self.model.add_parameters((8, 6))
    self.b_term1 = self.model.add_parameters((8))
    self.W_term = self.model.add_parameters((1, 8))

    self.W_final = self.model.add_parameters((1, 6))

  def Save(self, filename):
    self.model.save(filename)

  def Load(self, filename):
    self.model.populate(filename)

  def idf_val(self, w):
    if w in self.idf:
      return self.idf[w]
    return self.max_idf

  def query_doc_overlap(self, qwords, dwords):
    # % Query words in doc.
    qwords_in_doc = 0
    idf_qwords_in_doc = 0.0
    idf_qwords = 0.0
    for qword in utils.uwords(qwords):
      idf_qwords += self.idf_val(qword)
      for dword in utils.uwords(dwords):
        if qword == dword:
          idf_qwords_in_doc += self.idf_val(qword)
          qwords_in_doc += 1
          break
    if len(qwords) <= 0:
      qwords_in_doc_val = 0.0
    else:
      qwords_in_doc_val = (float(qwords_in_doc) /
                           float(len(utils.uwords(qwords))))
    if idf_qwords <= 0.0:
      idf_qwords_in_doc_val = 0.0
    else:
      idf_qwords_in_doc_val = float(idf_qwords_in_doc) / float(idf_qwords)

    # % Query bigrams  in doc.
    qwords_bigrams_in_doc = 0
    idf_qwords_bigrams_in_doc = 0.0
    idf_bigrams = 0.0
    for qword in utils.ubigrams(qwords):
      wrds = qword.split('_')
      idf_bigrams += self.idf_val(wrds[0]) * self.idf_val(wrds[1])
      for dword in utils.ubigrams(dwords):
        if qword == dword:
          qwords_bigrams_in_doc += 1
          idf_qwords_bigrams_in_doc += (self.idf_val(wrds[0])
                                        * self.idf_val(wrds[1]))
          break
    if len(qwords) <= 0:
      qwords_bigrams_in_doc_val = 0.0
    else:
      qwords_bigrams_in_doc_val = (float(qwords_bigrams_in_doc) /
                                   float(len(utils.ubigrams(qwords))))
    if idf_bigrams <= 0.0:
      idf_qwords_bigrams_in_doc_val = 0.0
    else:
      idf_qwords_bigrams_in_doc_val = (float(idf_qwords_bigrams_in_doc) /
                                       float(idf_bigrams))

    return [qwords_in_doc_val,
            qwords_bigrams_in_doc_val,
            idf_qwords_in_doc_val,
            idf_qwords_bigrams_in_doc_val]

  def get_words(self, s):
    sl = utils.tokenize(s)
    sl = [s for s in sl]
    return sl

  def GetExtraFeatures(self, qtext, dtext, bm25):
    qwords = self.get_words(qtext)
    dwords = self.get_words(dtext)
    qd1 = self.query_doc_overlap(qwords, dwords)
    bm25 = [bm25]
    return qd1 + bm25

  def Cosine(self, v1, v2):
    return dy.cdiv(dy.dot_product(v1, v2),
                   dy.l2_norm(v1) * dy.l2_norm(v2))

  def SetDropout(self, val):
    if val > 0.0:
      self.biRNN.set_dropout(val)
    else:
      self.biRNN.disable_dropout()

  def MakeInputs(self, text, train=False):
    words = self.get_words(text)
    wds = []
    vecs = []
    for w in words:
      if w in self.wv:
        vec = dy.inputVector(self.wv[w])
        vecs.append(dy.nobackprop(vec))
        wds.append(w)
    cont_sen = self.biRNN.transduce([self.pad[0]] +
                                    vecs +
                                    [self.pad[1]])[1:-1]
    cont_sen = [dy.esum([v, c]) for v, c in zip(vecs, cont_sen)]
    return wds, vecs, cont_sen

  def GetPOSIT(self, qvecs, sims, w2v_sims, matches):
    qscores = []
    for qtok in range(len(qvecs)):
      # Basic matches, max-sim, average-kmax-sim, exact match
      svec = dy.concatenate(sims[qtok])
      sim = dy.kmax_pooling(dy.transpose(svec), 1)[0]
      sim5 = dy.mean_elems(dy.kmax_pooling(dy.transpose(svec), 5)[0])
      wvec = dy.concatenate(w2v_sims[qtok])
      wsim = dy.kmax_pooling(dy.transpose(wvec), 1)[0]
      wsim5 = dy.mean_elems(dy.kmax_pooling(dy.transpose(wvec), 5)[0])
      mvec = dy.concatenate(matches[qtok])
      msim = dy.kmax_pooling(dy.transpose(mvec), 1)[0]
      msim5 = dy.mean_elems(dy.kmax_pooling(dy.transpose(mvec), 5)[0])
      layer1 = (self.W_term1.expr() *
                dy.concatenate(
                    [sim, sim5,
                     wsim, wsim5,
                     msim, msim5
                    ]) +
                self.b_term1.expr())
      qscores.append(self.W_term.expr() * utils.leaky_relu(layer1))

    return qscores

  def GetQDScore(self, qwds, qw2v, qvecs, dwds, dw2v, dvecs, extra,
                 train=False):
    nq = len(qvecs)
    nd = len(dvecs)
    qgl = [self.W_gate.expr() *
           dy.concatenate([qv, dy.constant(1, self.idf_val(qw))])
           for qv, qw in zip(qvecs, qwds)]
    qgates = dy.softmax(dy.concatenate(qgl))

    sims = []
    for qv in qvecs:
      dsims = []
      for dv in dvecs:
        dsims.append(self.Cosine(qv, dv))
      sims.append(dsims)

    w2v_sims = []
    for qv in qw2v:
      dsims = []
      for dv in dw2v:
        dsims.append(self.Cosine(qv, dv))
      w2v_sims.append(dsims)

    matches = []
    for qw in qwds:
      dmatch = []
      for dw in dwds:
        dmatch.append(dy.ones(1) if qw == dw else dy.zeros(1))
      matches.append(dmatch)

    qscores = self.GetPOSIT(qvecs, sims, w2v_sims, matches)

    # Final scores and ultimate classifier.
    qterm_score = dy.dot_product(dy.concatenate(qscores), qgates)

    fin_score = (self.W_final.expr() * dy.concatenate([qterm_score,
                                                       extra]))
    return fin_score

  def PairAppendToLoss(self, pos, neg, loss):
    if len(pos) != 1:
      print('ERROR IN POS EXAMPLE SIZE')
      print(len(pos))
    loss.append(dy.hinge(dy.concatenate(pos + neg), 0))


  def UpdateBatch(self, loss):
    if len(loss) > 0:
      sum_loss = dy.esum(loss)
      sum_loss.scalar_value()
      sum_loss.backward()
      self.trainer.update()
    dy.renew_cg()
