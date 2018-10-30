# POSIT-DRMM
Run DRMM POSIT model on the BioASQ data:
```
python3 drmm_main.py --dynet-mem 2000
```
Models and output predictions on the dev set are dumped after every iteration over the data.

Let's say on epoch 5 you have the best dev accuracy. This can be tested by going into the top-level eval/ directory and running
```
python3 run_eval.py ../bioasq_data/bioasq.dev.json ../models/drmm/posit_5.json | egrep '^map |^P_20 |^ndcg_cut_20 '
```
Where posit_5.json is the output predictions for the dev data on the 5th iteration.

To get the test predictions, run:
```
python3 drmm_predict.py model_posit_ep5 test --dynet-mem 2000
```
To evaluate the scores, run:
```
python3 run_eval.py ../bioasq_data/bioasq.test.json ../models/drmm/posit_test_preds.json | egrep '^map |^P_20 |^ndcg_cut_20 '
```

