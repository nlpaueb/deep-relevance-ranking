# Deep Relevance Ranking Using Enhanced Document-Query Interactions

This software accompanies  the following paper:
>R. McDonald, G. Brokos and I. Androutsopoulos, "Deep Relevance Ranking Using Enhanced Document-Query Interactions". Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2018), Brussels, Belgium, 2018. [[PDF](http://nlp.cs.aueb.gr/pubs/emnlp2018.pdf)], [[appendix](http://nlp.cs.aueb.gr/pubs/emnlp2018_appendix.pdf)]

It contains the code of the deep relevance ranking models described in the paper, which can be used to rerank the top-k documents returned by a BM25 based search engine.

# Instructions
This is a Python 3.6 project.

**Step 1**: Install the required Python packages: 

```
pip3 install -r requirements.txt
```

**Step 2**: Download the dataset(s) you intend to use (BioASQ and/or TREC ROBUST2004). 

```
sh get_bioasq_data.sh
sh get_robust04_data.sh
```

For each dataset, the following data are provided (among other files):

* Top-k documents retrieved by a BM25 based search engine ([Galago](http://www.lemurproject.org/galago.php)) for each query of the corresponding dataset.
* Pre-trained word embeddings
* IDF values

*Note: Downloading time may vary depending on server availability.*

**Step 3**: Navigate to a models directory to train the specific model and evaluate its performance on the test set. E.g. navigate to the PACRR (and PACRR-DRMM) model:
```
cd models/pacrr
```
Consult the README file of each model for dedicated instructions (e.g. [instructions for PACRR](https://github.com/nlpaueb/deep-relevance-ranking/tree/master/models/pacrr#pacrr)).
