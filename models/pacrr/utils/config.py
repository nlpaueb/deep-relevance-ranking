# Original code authors: Kai Hui and Andrew Yates
# This is a copy of the code of Kai Hui and Andrew Yates, accompanying the following papers:
# Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo. PACRR: A Position-Aware Neural IR Model for Relevance Matching. In EMNLP, 2017.
# Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo. Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval. In WSDM, 2018.
# Github repository of the original code: https://github.com/khui/copacrr

import datetime
import getpass
import itertools
import os
import sys
import time
all_years = ['09', '10', '11', '12', '13', '14']
#train_test_years = {'wt12_13':['wt11', 'wt14']}
train_test_years = {'wt' + '_'.join(sorted(years)):
        sorted(['wt' + ty for ty in all_years if ty not in years])
        for years in itertools.combinations(all_years, 4)}


# modelfile-modelname
file2name = {'pacrr':'PACRR', 'matchpyramid':'MatchPyramid', 'duetl':'DUETL',
             'drmm':'DRMM', 'knrm':'KNRM', 'elm_pacrr': 'ELMPACRR'}
name2file = {v: k for k, v in file2name.items()}

pipeline_params = ['expname', 'modelfn', 'train_years', 'seed', 'parentdir']

param2acronym=(('POS_METHOD',''), ('BINARY',''), ('QPROXIMITY', 'qp'), \
        ('CONTEXT', 'ct'), \
        ('PERMUTE', 'pm'), ('COMBINE', 'ff'), ('MORE_FILTERS',''), ('CASCADE_POS',''),\
        ('SIM_DIM','sd'), ('WIN_LEN','wl'),('NUM_NEG','nn'), ('n_batch','nb'))



# concat of different qrel from trec
# please download from http://trec.nist.gov/data/webmain.html
# for instance: http://trec.nist.gov/data/web/2014/qrels.adhoc.txt
cur_dir=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
qrelfdir=os.path.join(cur_dir, "data")
qrelf=os.path.join(qrelfdir,"qrels.adhoc.6y")
# the evaluation script from trec:
# http://trec.nist.gov/data/web/12/gdeval.pl
perlf=os.path.join(cur_dir,"libs","gdeval.pl")
# http://trec.nist.gov/trec_eval/trec_eval_latest.tar.gz, please compile after unpacking 
treceval=os.path.join(cur_dir,"libs","trec_eval.9.0","trec_eval")

'''
simmat directory and its structure.
1) simmat for topic field in the trec query
rawdoc_mat_dir/topic_doc_mat/qid/clueweb_id.npy
like:
rawdoc_mat_dir/topic_doc_mat/1/clueweb09-en0132-08-08589.npy

2) simmat for description field in the trec query
rawdoc_mat_dir/desc_doc_mat/qid/clueweb_id.npy
like:
rawdoc_mat_dir/desc_doc_mat/1/clueweb09-en0132-08-08589.npy

3) simmat for the idf of different terms
rawdoc_mat_dir/query_idf/desc_term_idf/qid.npy
rawdoc_mat_dir/query_idf/topic_term_idf/qid.npy
like:
rawdoc_mat_dir/query_idf/desc_term_idf/1.npy
rawdoc_mat_dir/query_idf/topic_term_idf/1.npy
'''
# the directory helds the similarity matrices 
sim_dir="/local/var/tmp/%s" % getpass.getuser()
rawdoc_mat_dir=os.path.join(sim_dir, 'cosine')
#"/directory to the pre-computed similarity matrices/"
# the mat for the context needs to be pre-computed if context=True
contextdir = None
if contextdir is None:
    contextdir = "%s/context" % rawdoc_mat_dir

# following filename is used in evals/*.py
# all runs from trec for the rerank-simple and rerank-all benchmarks 
trec_run_basedir=os.path.join(cur_dir, "data", "trec_runs")
# the evaluation results for all origial runs 
eval_trec_run_basedir=os.path.join(cur_dir, "data", "eval_trec_runs")



def default_params():
    expname = "debug" # experiment name
    modelfn = "pacrr" # model to run
    train_years = "wt09_10_11_12" # years to train on
    test_year = "wt13" # year to predict on
    seed = 7541
    parentdir = '/user/%s'% getpass.getuser()
    outdir = '/user/%s'% getpass.getuser()

    simdim = 800 # length of document dimension
    binmat = False # use binary similarity matrices? (boolean)
    numneg = 1 # number of non-relevant docs in softmax
    batch = 16 # batch size
    epochs = 150 # number of iterations to run
    nsamples = 2048 # samples per epoch
    maxqlen = 16 # maximum query length
    
    distill = "firstk" # similarity matrix distillation method
    nfilter = 32 # number of filters to use for the n-gram convolutions
    winlen = 3 # maximum n-gram length
    kmaxpool = 3 # top k for max pooling
    combine = 32 # type of combination layer to use. 0 for an LSTM, otherwise the number of feedforward layer dimensions
    qproximity = 0 # additional NxN proximity filter to include (0 to disable)
    context = True # include match contexts? (boolean)
    shuffle = True # shuffle input to the combination layer? (i.e., LSTM or feedforward layer)

    ek = 10 # topk expansion terms to use when enhance=qexpand or enhance=both
    
    
    # configure the sizes of extra filters with format: axb.cxd.<more>
    # for example: 1x100.3x1 => [(1,100), (3,1)]
    # to turn off, set to an empty string
    xfilters = ""

    # configure the cascade mode of the max-pooling
    # Namely, pool over [first10, first20, ..., first100, wholedoc]
    # instead of only pool on complete document
    # the input is a list of relative positions for pooling
    # for example, 25.50.75.100 => [25,50,75,100] on a doc with length 100
    # to turn off, set to an empty string
    cascade = ""

    # (DRMM only) size of first dense layer
    drmmdense = 5

    ut = True # use topics in queries
    ud = True # use descriptions in queries

    if 'debug' in expname:
        epochs = 5
