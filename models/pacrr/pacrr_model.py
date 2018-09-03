# Original code authors: Kai Hui and Andrew Yates
# This is a modified version of the code of Kai Hui and Andrew Yates. The original code accompanies the following papers:
# Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo. PACRR: A Position-Aware Neural IR Model for Relevance Matching. In EMNLP, 2017.
# Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo. Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval. In WSDM, 2018.
# GitHub repository of the original code: https://github.com/khui/copacrr

from keras.models import Sequential, Model
from keras.layers import Permute, Activation, Dense, Dropout, Embedding, \
Flatten, Input, merge, Lambda, Reshape, Convolution2D, MaxPooling2D, Bidirectional, Dot, TimeDistributed, Multiply
from keras.layers.merge import Concatenate
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import Adam
from keras import backend
from model_base import MODEL_BASE
import tensorflow as tf
from utils.ngram_nfilter import get_ngram_nfilter
import keras.backend as K

class PACRR(MODEL_BASE):
    
    params = MODEL_BASE.common_params + ['distill', 'winlen', 'nfilter', 'kmaxpool', 'combine',
                                         'qproximity', 'context', 'shuffle', 'xfilters', 'cascade']

    def __init__(self, *args, **kwargs):
        super(PACRR, self).__init__(*args, **kwargs)
        self.NGRAM_NFILTER, _ = get_ngram_nfilter(self.p['winlen'], self.p['qproximity'],
                                                  self.p['maxqlen'], self.p['xfilters'])
        self.NGRAMS = sorted(self.NGRAM_NFILTER.keys())
        if self.p['qproximity'] > 0:
            self.NGRAMS.append(self.p['qproximity'])
                
    def _cascade_poses(self):
        '''
        initialize the cascade positions, over which
        we max-pool after the cnn filters.
        the outcome is a list of document positions.
        when the list only includes the SIM_DIM, it 
        is equivalent to max-pool over the whole document
        '''
        doc_poses = list()
        pos_arg = str(self.p['cascade'])
        if len(pos_arg) > 0:
            poses = pos_arg.split('.')
            for p in poses:
                if len(p) > 0:
                    p = int(p)
                    if p <= 0 or p > 100:
                        raise ValueError("Cascade positions are outside (0,100]: %s"%pos_arg)
            doc_poses.extend([int((int(p)/100)*self.p['simdim']) for p in poses if len(p)>0])

        if self.p['simdim'] not in doc_poses:
            doc_poses.append(self.p['simdim'])
            
        return doc_poses

    def build_doc_scorer(self, r_query_idf, permute_idxs):
        p = self.p
        ng_fsizes = self.NGRAM_NFILTER

        maxpool_poses = self._cascade_poses()

        filter_sizes = list()
        added_fs = set()
        for ng in sorted(ng_fsizes):
            # n-gram in input
            for n_x, n_y in ng_fsizes[ng]:
                dim_name = self._get_dim_name(n_x, n_y)
                if dim_name not in added_fs:
                    filter_sizes.append((n_x,n_y)) 
                    added_fs.add(dim_name)

        re_input, cov_sim_layers, pool_sdim_layer, pool_sdim_layer_context, pool_filter_layer, ex_filter_layer, re_lq_ds =\
        self._cov_dsim_layers(p['simdim'], p['maxqlen'], filter_sizes, p['nfilter'], top_k=p['kmaxpool'], poses=maxpool_poses, selecter=p['distill'])

        query_idf = Reshape((p['maxqlen'], 1))(Activation('softmax',
                            name='softmax_q_idf')(Flatten()(r_query_idf)))


        if p['combine'] < 0:
            raise RuntimeError("combine should be 0 (LSTM) or the number of feedforward dimensions")
        elif p['combine'] == 0:
            rnn_layer = LSTM(1, dropout=0.0, recurrent_regularizer=None, recurrent_dropout=0.0, unit_forget_bias=True, \
                    name="lstm_merge_score_idf", recurrent_activation="hard_sigmoid", bias_regularizer=None, \
                    activation="tanh", recurrent_initializer="orthogonal", kernel_regularizer=None, kernel_initializer="glorot_uniform")

        else:
            if not p['td']:
                dout = Dense(1, name='dense_output')
                d1 = Dense(p['combine'], activation='relu', name='dense_1')
                d2 = Dense(p['combine'], activation='relu', name='dense_2')
                rnn_layer = lambda x: dout(d1(d2(x)))
            else:
                dout_combine = Dense(1, name='q_scores_dense_output', use_bias=True)
                dout = Dense(1, name='dense_output')
                d1 = Dense(p['combine'], activation='relu', name='dense_1')
                d2 = TimeDistributed(Dense(p['combine'], activation='relu', name='dense_2'), input_shape=(p['maxqlen'], p['kmaxpool']*p['winlen']+1))
                rnn_layer = lambda x: dout(d1(d2((x))))

        combine_scores = Dense(1, name='final_dense_output', use_bias=False)

        def _permute_scores(inputs):
            scores, idxs = inputs
            return tf.gather_nd(scores, backend.cast(idxs, 'int32'))


        self.vis_out = None
        self.visout_count = 0
        def _scorer(doc_inputs, dataid):
            self.visout_count += 1
            self.vis_out = {}
            doc_qts_scores = [query_idf]
            for ng in sorted(ng_fsizes):
                if p['distill'] == 'firstk':
                    input_ng = max(ng_fsizes)
                else:
                    input_ng = ng
                    
                for n_x, n_y in ng_fsizes[ng]:
                    dim_name = self._get_dim_name(n_x, n_y)
                    if n_x == 1 and n_y == 1:
                        doc_cov = doc_inputs[input_ng]
                        re_doc_cov = doc_cov
                    else:
                        doc_cov = cov_sim_layers[dim_name](re_input(doc_inputs[input_ng]))
                        re_doc_cov = re_lq_ds[dim_name](pool_filter_layer[dim_name](Permute((1, 3, 2))(doc_cov)))
                    self.vis_out['conv%s' % ng] = doc_cov
                        
                    if p['context']:
                        ng_signal = pool_sdim_layer_context[dim_name]([re_doc_cov, doc_inputs['context']])
                    else:
                        ng_signal = pool_sdim_layer[dim_name](re_doc_cov)
                    
                    doc_qts_scores.append(ng_signal)

            if len(doc_qts_scores) == 1:                
                doc_qts_score = doc_qts_scores[0]
            else:
                doc_qts_score = Concatenate(axis=2)(doc_qts_scores)

            if permute_idxs is not None:
                doc_qts_score = Lambda(_permute_scores)([doc_qts_score, permute_idxs])

            if not p['td']:
                # Original PACRR architecture
                doc_qts_score = Flatten()(doc_qts_score)
                if p['use_bm25']:
                    doc_qts_score = Concatenate(axis=1)([doc_qts_score, doc_inputs['bm25_score']])
                if p['use_overlap_features']:
                    doc_qts_score = Concatenate(axis=1)([doc_qts_score, doc_inputs['doc_overlap_features']])
                doc_score = rnn_layer(doc_qts_score)
            else:
                # PACRR-DRMM architecture
                doc_score = Flatten()(rnn_layer(doc_qts_score))
                if p['use_bm25']:
                    doc_score = Concatenate(axis=1)([doc_score, doc_inputs['bm25_score']])
                if p['use_overlap_features']:
                    doc_score = Concatenate(axis=1)([doc_score, doc_inputs['doc_overlap_features']])
                doc_score = dout_combine(doc_score)

            return doc_score

        return _scorer

    def _create_inputs(self, prefix, q_emb_mat, embeddings):
        p = self.p
        if p['distill'] == 'firstk':
            ng = max(self.NGRAMS)

            doc_inds = Input(shape = (p['simdim'],), name='{0}_doc_inds'.format(prefix))
            doc_bm25_score = Input(shape=(1,), name='{0}_doc_bm25_scores'.format(prefix))
            doc_overlap_features = Input(shape=(3,), name='{0}_doc_overlap_vec'.format(prefix))

            d_emb_mat = embeddings(doc_inds)

            shared = Dot(axes=(2, 2), normalize=True)([q_emb_mat, d_emb_mat])
            
            inputs = {ng: shared, 'bm25_score': doc_bm25_score, 'doc_overlap_features': doc_overlap_features}
        else:
            inputs = {}
            for ng in self.NGRAMS:
                inputs[ng] = Input(shape = (p['maxqlen'], p['simdim']), name='%s_wlen_%d' % (prefix, ng))
            
        return inputs, doc_inds, doc_bm25_score, doc_overlap_features
    

    def build(self):
        p = self.p

        r_query_idf = Input(shape = (p['maxqlen'], 1), name='query_idf')
        
        if p['shuffle']:
            permute_input = Input(shape=(p['maxqlen'], 2), name='permute', dtype='int32')
        else:
            permute_input = None

        embeddings = Embedding(p['vocab_size'], p['emb_dim'], weights=[p['embed']], trainable=False, mask_zero=True)

        query_inds = Input(shape = (p['maxqlen'],), name='query_inds')
        query_emb_mat = embeddings(query_inds)

        doc_scorer = self.build_doc_scorer(r_query_idf, permute_idxs=permute_input)

        pos_inputs, pos_doc_inds, pos_bm25_score, pos_overlap_vec = self._create_inputs('pos', query_emb_mat, embeddings)

        if p['context']:
            pos_inputs['context'] = Input(shape=(p['maxqlen'], p['simdim']), name='pos_context')
            
        neg_inputs = {}
        for neg_ind in range(p['numneg']):
            neg_inputs[neg_ind], neg_doc_inds, neg_bm25_score, neg_overlap_vec = self._create_inputs('neg%d' % neg_ind, query_emb_mat, embeddings)
            if p['context']:
                neg_inputs[neg_ind]['context'] = Input(shape=(p['maxqlen'], p['simdim']),
                                                       name='neg%d_context' % neg_ind)

        pos_score = doc_scorer(pos_inputs, 'pos')
        neg_scores = [doc_scorer(neg_inputs[neg_ind], 'neg_%s'%neg_ind) for neg_ind in range(p['numneg'])]

        pos_neg_scores = [pos_score] + neg_scores
        pos_prob = Lambda(self.pos_softmax, name='pos_softmax_loss')(pos_neg_scores)
        
        pos_input_list = [pos_doc_inds, pos_bm25_score, pos_overlap_vec]
        neg_input_list = [neg_doc_inds, neg_bm25_score, neg_overlap_vec]
        inputs = [query_inds] + pos_input_list + neg_input_list + [r_query_idf]
        if p['shuffle']:
            inputs.append(permute_input)
        
        self.model = Model(inputs = inputs, outputs = [pos_prob])

        self.scoring_model = Model(inputs = [query_inds] + pos_input_list + [r_query_idf], outputs = [pos_score])

        self.model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

        return self.model, self.scoring_model

    def build_predict(self):
        p = self.p
        r_query_idf = Input(shape = (p['maxqlen'], 1), name='query_idf')
        if p['shuffle']:
            permute_input = Input(shape=(p['maxqlen'], 2), name='permute', dtype='int32')
        else:
            permute_input = None

        embeddings = Embedding(p['vocab_size'], p['emb_dim'], trainable=False, mask_zero=True)

        query_inds = Input(shape = (p['maxqlen'],), name='query_inds')
        query_emb_mat = embeddings(query_inds)

        doc_scorer = self.build_doc_scorer(r_query_idf, permute_idxs=permute_input)

        pos_inputs, pos_doc_inds, pos_bm25_score, pos_overlap_vec = self._create_inputs('pos', query_emb_mat, embeddings)

        if p['context']:
            pos_inputs['context'] = Input(shape=(p['maxqlen'], p['simdim']), name='pos_context')

        pos_score = doc_scorer(pos_inputs, 'pos')
        pos_input_list = [pos_doc_inds, pos_bm25_score, pos_overlap_vec]

        inputs = [query_inds] + pos_input_list + [r_query_idf]
        if p['shuffle']:
            inputs.append(permute_input)

        self.scoring_model = Model(inputs = [query_inds] + pos_input_list + [r_query_idf], outputs = [pos_score])
        
        return self.scoring_model
