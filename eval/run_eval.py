import os
import sys
import json
import uuid
import datetime
import subprocess


def format_bioasq2treceval_qrels(bioasq_data, filename):
	with open(filename, 'w') as f:
		for q in bioasq_data['questions']:
			for d in q['documents']:
				f.write('{0} 0 {1} 1'.format(q['id'], d))
				f.write('\n')

def format_bioasq2treceval_qret(bioasq_data, system_name, filename):
	with open(filename, 'w') as f:
		for q in bioasq_data['questions']:
			rank = 1
			for d in q['documents']:
				sim = (len(q['documents']) + 1 - rank) / float(len(q['documents'])) # Just a fake similarity. Does not affect the metrics we are using.
				f.write('{0} {1} {2} {3} {4} {5}'.format(q['id'], 0, d, rank, sim, system_name))
				f.write('\n')
				rank += 1

def trec_evaluate(qrels_file, qret_file):
	trec_eval_res = subprocess.Popen(
		[os.path.dirname(os.path.realpath(__file__)) + '/trec_eval.9.0/./trec_eval', '-m', 'all_trec', qrels_file, qret_file],
		stdout=subprocess.PIPE, shell=False)

	(out, err) = trec_eval_res.communicate()
	trec_eval_res = out.decode("utf-8")
	print(trec_eval_res)

if __name__ == '__main__':

	try:
		golden_file = sys.argv[1]
		predictions_file = sys.argv[2]
	except:
		sys.exit("Provide golden and predictions files.")
	
	try:
		system_name = sys.argv[3]
	except :
		try:
			system_name = predictions_file.split('/')[-1]
		except:
			system_name = predictions_file

	with open(golden_file, 'r') as f:
		golden_data = json.load(f)

	with open(predictions_file, 'r') as f:
		predictions_data = json.load(f)

	temp_dir = uuid.uuid4().hex
	qrels_temp_file = '{0}/{1}'.format(temp_dir, 'qrels.txt')
	qret_temp_file = '{0}/{1}'.format(temp_dir, 'qret.txt')

	try:
		if not os.path.exists(temp_dir):
			os.makedirs(temp_dir)
		else:
			sys.exit("Possible uuid collision")

		format_bioasq2treceval_qrels(golden_data, qrels_temp_file)
		format_bioasq2treceval_qret(predictions_data, system_name, qret_temp_file)

		trec_evaluate(qrels_temp_file, qret_temp_file)
	finally:
		os.remove(qrels_temp_file)
		os.remove(qret_temp_file)
		os.rmdir(temp_dir)

