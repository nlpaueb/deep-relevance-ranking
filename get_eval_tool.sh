#!/bin/bash

cd eval

wget -c https://trec.nist.gov/trec_eval/trec_eval_latest.tar.gz
tar -zxvf trec_eval_latest.tar.gz

cd trec_eval.9.0
make

