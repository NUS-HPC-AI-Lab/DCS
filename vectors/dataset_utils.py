""" We follow existing works to format prompts and generate output probabilities for
	classification datasets including sst5, agnews, dbpedia, rte, and trec
"""
import os
import re
import pandas as pd
import json
import random
import ast
import pickle
import numpy as np
import datasets
from datasets import load_dataset


def preproc_input(sent):
	preproc_sent = ' '.join(re.findall(r'@[\w$]+|\w+|[^ \t\n]', sent)).lower()
	return preproc_sent

def sample_subset(sentences, labels, num_samples=None, max_length=None, rseed=0):
	"""for train dataset larger than 10k samples, 
	we randomly sample a training subset of 10k"""
	if max_length is not None:
		out_sentences = []
		out_labels = []
		for idx in range(len(sentences)):
			if len(sentences[idx]) <= max_length:
				out_sentences.append(sentences[idx])
				out_labels.append(labels[idx])
		sentences = out_sentences
		labels = out_labels

	if num_samples is not None:
		np.random.seed(rseed)
		inds = np.random.choice(len(labels), size=num_samples, replace=False)
		selected_sentences = [sentences[i] for i in inds]
		selected_labels = [labels[i] for i in inds]
	else:
		selected_sentences, selected_labels = sentences, labels

	return selected_sentences, selected_labels

def sample_subset_by_lbl(vec_dir, num_class, sentences, labels, preds_not_used, num_samples=None, rseed=0):
	if num_samples is not None:
		if num_samples <= 100:
			np.random.seed(rseed)
			all_class_samples_inds = {k: [] for k in range(num_class)}
			for i in range(len(labels)):
				all_class_samples_inds[labels[i]].append(i)

			all_selected_sentences = []
			all_selected_labels = []
			all_selected_preds_not_used = []
			# selected_class_samples_inds = {k: [] for k in range(num_class)}
			num_sample_class = {k: int(num_samples/num_class) for k in range(num_class-1)}
			num_sample_class[num_class-1] = num_samples - int(num_samples/num_class)*(num_class-1)
			for k in all_class_samples_inds:
				inds = np.random.choice(all_class_samples_inds[k], size=num_sample_class[k], replace=False)
				selected_sentences = [sentences[i] for i in inds]
				all_selected_sentences.extend(selected_sentences)
				selected_labels = [labels[i] for i in inds]
				all_selected_labels.extend(selected_labels)
				selected_preds_not_used = [preds_not_used[i] for i in inds]
				all_selected_preds_not_used.extend(selected_preds_not_used)

			random.seed(rseed)
			combined_lists = list(zip(all_selected_sentences, all_selected_labels, all_selected_preds_not_used))
			random.shuffle(combined_lists)
			final_selected_sentences, final_selected_labels, final_selected_preds_not_used = zip(*combined_lists)
			assert len(final_selected_sentences) == num_samples
		else:
			np.random.seed(rseed)
			inds = np.random.choice(len(labels), size=num_samples, replace=False)
			final_selected_sentences = [sentences[i] for i in inds]
			final_selected_labels = [labels[i] for i in inds]
			final_selected_preds_not_used = [preds_not_used[i] for i in inds]
		
		# save to files
		res_dir = os.path.join(vec_dir, f"train_{num_samples}")
		if not os.path.exists(res_dir):
			 os.makedirs(res_dir)
		if res_dir is not None:
			res_fp = os.path.join(res_dir, os.path.join("train.txt"))
			if not os.path.exists(res_fp):
				save_selected_vecs_txt(res_fp, final_selected_sentences, final_selected_labels, final_selected_preds_not_used)

	else:
		final_selected_sentences, final_selected_labels, final_selected_preds_not_used = sentences, labels, preds_not_used

	return final_selected_sentences, final_selected_labels, final_selected_preds_not_used

def updown_sample_subset_by_lbl(vec_dir, num_class, sentences, labels, preds_not_used, num_samples=None, rseed=0):
	if num_samples is not None:
		np.random.seed(rseed)
		all_class_samples_inds = {k: [] for k in range(num_class)}
		for i in range(len(labels)):
			all_class_samples_inds[labels[i]].append(i)

		all_selected_sentences = []
		all_selected_labels = []
		all_selected_preds_not_used = []
		# selected_class_samples_inds = {k: [] for k in range(num_class)}
		num_sample_class = {k: int(num_samples/num_class) for k in range(num_class-1)}
		num_sample_class[num_class-1] = num_samples - int(num_samples/num_class)*(num_class-1)
		for k in all_class_samples_inds:
			if len(all_class_samples_inds[k]) >= num_sample_class[k]:
				inds = np.random.choice(all_class_samples_inds[k], size=num_sample_class[k], replace=False)
			else:
				# Upsample to get extra samples
				inds_extra = np.random.choice(all_class_samples_inds[k], size=num_sample_class[k]-len(all_class_samples_inds[k]), replace=True)
				# print('all_class_samples_inds[k]', all_class_samples_inds[k])
				# print('inds_extra', list(inds_extra))
				inds = all_class_samples_inds[k] + list(inds_extra)
				assert len(inds) == num_sample_class[k]
			selected_sentences = [sentences[i] for i in inds]
			all_selected_sentences.extend(selected_sentences)
			selected_labels = [labels[i] for i in inds]
			all_selected_labels.extend(selected_labels)
			selected_preds_not_used = [preds_not_used[i] for i in inds]
			all_selected_preds_not_used.extend(selected_preds_not_used)

		random.seed(rseed)
		combined_lists = list(zip(all_selected_sentences, all_selected_labels, all_selected_preds_not_used))
		random.shuffle(combined_lists)
		final_selected_sentences, final_selected_labels, final_selected_preds_not_used = zip(*combined_lists)
		assert len(final_selected_sentences) == num_samples
		
		# save to files
		res_dir = os.path.join(vec_dir, f"train_{num_samples}")
		if not os.path.exists(res_dir):
			 os.makedirs(res_dir)
		if res_dir is not None:
			res_fp = os.path.join(res_dir, os.path.join("opt.txt"))
			if not os.path.exists(res_fp):
				save_selected_vecs_txt(res_fp, final_selected_sentences, final_selected_labels, final_selected_preds_not_used)

	else:
		final_selected_sentences, final_selected_labels, final_selected_preds_not_used = sentences, labels, preds_not_used

	return final_selected_sentences, final_selected_labels, final_selected_preds_not_used


def save_selected_vecs_txt(res_fp, sentences, labels, preds):
	with open(res_fp, 'w') as f:
		for i in range(len(labels)):
			class_prob_i = ' '.join([format(x, '.6f') for x in sentences[i]])
			f.write(str(i)+' '+str(labels[i])+' '+str(preds[i])+' '+str(class_prob_i)+'\n')

def load_sst5():
	train_dataset = load_dataset("SetFit/sst5", split="train")
	test_dataset = load_dataset("SetFit/sst5", split="test")
	return train_dataset['text'], train_dataset['label'], test_dataset['text'], test_dataset['label']

def load_agnews():
	train_data = pd.read_csv('data/agnews/train.csv')
	test_data = pd.read_csv('data/agnews/test.csv')

	train_sentences = train_data['Title'] + ". " + train_data['Description']
	train_sentences = list(
		[item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
		 in train_sentences]) # some basic cleaning
	train_labels = list(train_data['Class Index'])
	test_sentences = test_data['Title'] + ". " + test_data['Description']
	test_sentences = list(
		[item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
		 in test_sentences]) # some basic cleaning
	test_labels = list(test_data['Class Index']) 
	train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
	test_labels = [l - 1 for l in test_labels]

	# keep a train subset of 10k samples
	train_sentences_10k, train_labels_10k = sample_subset(train_sentences, train_labels, 10000, max_length=None, rseed=0)
	
	return train_sentences_10k, train_labels_10k, test_sentences, test_labels

def load_trec():
	inv_label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
	train_sentences = []
	train_labels = []
	with open('data/trec/train.txt', 'r') as train_data:
		for line in train_data:
			train_label = line.split(' ')[0].split(':')[0]
			train_label = inv_label_dict[train_label]
			train_sentence = ' '.join(line.split(' ')[1:]).strip()
			# basic cleaning
			train_sentence = train_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
			train_labels.append(train_label)
			train_sentences.append(train_sentence)

	test_sentences = []
	test_labels = []
	with open('data/trec/test.txt', 'r') as test_data:
		for line in test_data:
			test_label = line.split(' ')[0].split(':')[0]
			test_label = inv_label_dict[test_label]
			test_sentence = ' '.join(line.split(' ')[1:]).strip()
			test_sentence = test_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
			test_labels.append(test_label)
			test_sentences.append(test_sentence)
	return train_sentences, train_labels, test_sentences, test_labels

def load_dbpedia():
	train_data = pd.read_csv('data/dbpedia/train_subset.csv')
	test_data = pd.read_csv('data/dbpedia/test.csv')

	train_sentences = train_data['Text']
	train_sentences = list([item.replace('""', '"') for item in train_sentences])
	train_labels = list(train_data['Class'])

	test_sentences = test_data['Text']
	test_sentences = list([item.replace('""', '"') for item in test_sentences])
	test_labels = list(test_data['Class'])
	
	train_labels = [l - 1 for l in train_labels] # start from 0
	test_labels = [l - 1 for l in test_labels]

	# keep a train subset of 10k samples
	train_sentences_10k, train_labels_10k = sample_subset(train_sentences, train_labels, 10000, max_length=None, rseed=0)
	
	return train_sentences_10k, train_labels_10k, test_sentences, test_labels

def load_ddi():
	inv_label_dict = {'0': 0, 'DDI-effect': 1, 'DDI-mechanism': 2, 'DDI-advise': 3, 'DDI-int': 4}
	train_sentences = []
	train_labels = []
	with open('data/ddi/train.jsonl', 'r') as train_data:	# this is the 10k subset of DDI
		for line in train_data:
			myjson = json.loads(line)
			raw_sent = myjson['TEXT1']
			train_label = myjson['LBL']
			train_label = inv_label_dict[train_label]
			train_sentence = preproc_input(raw_sent)
			train_labels.append(train_label)
			train_sentences.append(train_sentence)

	test_sentences = []
	test_labels = []
	with open('data/ddi/test.jsonl', 'r') as test_data:
		for line in test_data:
			myjson = json.loads(line)
			raw_sent = myjson['TEXT1']
			test_label = myjson['LBL']
			test_label = inv_label_dict[test_label]
			test_sentence = preproc_input(raw_sent)
			test_labels.append(test_label)
			test_sentences.append(test_sentence)
	return train_sentences, train_labels, test_sentences, test_labels


def load_rte():
	train_questions = []
	train_answers = []
	with open("data/rte/train.jsonl", "r") as f:
		for line in f:
			myjson = json.loads(line)
			q = myjson['hypothesis']
			p = myjson['premise']
			if myjson['label'] == 'not_entailment':
				train_answers.append(0)
			elif myjson['label'] == 'entailment':
				train_answers.append(1)
			else:
				exit('answer')
			train_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

	test_questions = []
	test_answers = []
	with open("data/rte/val.jsonl", "r") as f:	# rte test set does not contain groundtruths
		for line in f:
			myjson = json.loads(line)
			q = myjson['hypothesis']
			p = myjson['premise']
			if myjson['label'] == 'not_entailment':
				test_answers.append(0)
			elif myjson['label'] == 'entailment':
				test_answers.append(1)
			else:
				exit('answer')
			test_questions.append(p + '\n' + 'question: ' + q + ' True or False?')
	
	return train_questions, train_answers, test_questions, test_answers

def load_pubmedqa():
	train_questions = []
	train_answers = []
	inv_label_dict = {'yes': 0, 'no': 1, 'maybe': 2}
	f = open("data/pubmedqa/ori_pqal.json", "r")
	train_data = json.load(f)
	for _, line in train_data.items():
		q = line['QUESTION']
		context = line['CONTEXTS']
		concat_c = '==='.join(context)
		ans = line['final_decision']
		train_answers.append(inv_label_dict[ans])
		train_questions.append('Context: ' + concat_c + '\n' + 'Question: ' + q + 'yes, no, or maybe?')

	test_questions = []
	test_answers = []
	f = open("data/pubmedqa/test_set.json", "r")
	test_data = json.load(f)
	for _, line in test_data.items():
		q = line['QUESTION']
		context = line['CONTEXTS']
		concat_c = '==='.join(context)
		ans = line['final_decision']
		test_answers.append(inv_label_dict[ans])
		test_questions.append('Context: ' + concat_c + '\n' + 'Question: ' + q + 'yes, no, or maybe?')

	return train_questions, train_answers, test_questions, test_answers

def read_dataset(params):
	if params['dataset'] == 'sst5':
		orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst5()
		params['prompt_prefix'] = ""
		params["q_prefix"] = "Review: "
		params["a_prefix"] = "Sentiment: "
		params['label_dict'] = {0: ['terrible'], 1: ['bad'], 2: ['okay'], 3: ['good'], 4: ['great']}
		params['task_format'] = 'classification'
		params['num_tokens_to_predict'] = 1

	elif params['dataset'] == 'agnews':
		orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()
		params['prompt_prefix'] = "Please classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
		params["q_prefix"] = "Article: "
		params["a_prefix"] = "Answer: "
		params['label_dict'] = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology']}
		params['task_format'] = 'classification'
		params['num_tokens_to_predict'] = 1

	elif params['dataset'] == 'trec':
		orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_trec()
		params['prompt_prefix'] = "Please classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
		params["q_prefix"] = "Question: "
		params["a_prefix"] = "Answer Type: "
		params['label_dict'] = {0: ['Number'], 1: ['Location'], 2: ['Person'], 3: ['Description'], 4: ['Entity'], 5: ['Ab']}
		params['task_format'] = 'classification'
		params['num_tokens_to_predict'] = 1

	elif params['dataset'] == 'rte':
		orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_rte()
		params['prompt_prefix'] = ""
		params["q_prefix"] = ""
		params["a_prefix"] = "answer: "
		params['label_dict'] = {0: ['False'], 1: ['True']}
		params['num_user_input'] = 2
		params['task_format'] = 'classification'
		params['num_tokens_to_predict'] = 1

	elif params['dataset'] == 'dbpedia':
		orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_dbpedia()
		params['prompt_prefix'] = "Please classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n"
		params["q_prefix"] = "Article: "
		params["a_prefix"] = "Answer: "
		params['label_dict'] = {0: ['Company'], 1: ['School'], 2: ['Artist'], 3: ['Ath'], 4: ['Polit'], 5: ['Transportation'], 6: ['Building'], 7: ['Nature'], 8: ['Village'], 9: ['Animal'], 10: ['Plant'], 11: ['Album'], 12: ['Film'], 13: ['Book']}
		params['task_format'] = 'classification'
		params['num_tokens_to_predict'] = 1
	
	elif params['dataset'] == 'ddi':
		orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ddi()
		params['prompt_prefix'] = "Please choose a most suitable answer from Negative, Effect, Mechanism, Advice, or Interaction, for the drug-drug interaction relation between the @drug$ pair in the following description.\n\n"
		params["q_prefix"] = "Description: "
		params["a_prefix"] = "Answer: "
		params['label_dict'] = {0: ['Negative'], 1: ['Effect'], 2: ['Mechanism'], 3: ['Advice'], 4: ['Interaction']}
		params['task_format'] = 'classification'
		params['num_tokens_to_predict'] = 1

	elif params['dataset'] == 'pubmedqa':
		orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_pubmedqa()
		params['prompt_prefix'] = "Please choose a most suitable answer from yes, no, or maybe, for the following question given a context.\n\n"
		params["q_prefix"] = ""
		params["a_prefix"] = "Answer: "
		params['label_dict'] = {0: ['yes'], 1: ['no'], 2: ['maybe']}
		params['task_format'] = 'classification'
		params['num_tokens_to_predict'] = 1
	
	else:
		raise NotImplementedError
	return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels









