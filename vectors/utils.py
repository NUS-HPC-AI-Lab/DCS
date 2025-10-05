import numpy as np
import time
from copy import deepcopy
import os
import sys
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")
from logging import raiseExceptions


def fix_seed_rerun(rseed):
	""" Enable reproducibility """
	torch.manual_seed(rseed)
	torch.cuda.manual_seed_all(rseed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(rseed)
	random.seed(rseed)
	os.environ['PYTHONHASHSEED'] = str(rseed)


def get_batches(prompts, bs):
	for i in range(0, len(prompts), bs):
		yield prompts[i:i + bs]


llama2_model = None
llama2_tokenizer = None


def setup_llama2(model_name, gpu_id):
	# load the Llama-2-13b model
	device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
	print('device', device)
	global llama2_model
	global llama2_tokenizer
	if llama2_model is None:
		print(f"Setting up {model_name} model")
		llama2_model = LlamaForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
		llama2_model.eval().to(device)
		llama2_tokenizer = LlamaTokenizer.from_pretrained(model_name)
		# to batch generation, we pad on the left and mask those positions out.
		llama2_tokenizer.padding_side = "left"
		llama2_tokenizer.pad_token = llama2_tokenizer.eos_token
		llama2_model.config.pad_token_id = llama2_model.config.eos_token_id
		print("Finished")
	return device


def construct_prompt(params, priming_sentences, priming_labels, infer_sentence):
	"""construct a 1-shot prompt to be fed into the model"""
	# take the prompt template and fill in the training and test example
	prompt = params["prompt_prefix"]
	q_prefix = params["q_prefix"]
	a_prefix = params["a_prefix"]
	for s, l in zip(priming_sentences, priming_labels):
		prompt += q_prefix
		prompt += s + "\n"
		if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64):
			assert params['task_format'] == 'classification'
			l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
		prompt += a_prefix
		prompt += l_str + "\n\n====\n\n"

	prompt += q_prefix
	prompt += infer_sentence + "\n"
	assert a_prefix[-1] == ' '
	prompt += a_prefix[:-1] # Remove trailing space
	return prompt


def get_class_probs(params, priming_sentences, priming_labels, infer_sentences, normalize=True):
	fix_seed_rerun(params['seed'])
	prompts = []
	for infer_sentence in infer_sentences:
		prompts.append(construct_prompt(params, priming_sentences, priming_labels, infer_sentence))
	batch_prompts = list(get_batches(prompts, params['bs']))

	all_raw_answers = []
	with torch.no_grad():
		# llama2
		if 'Llama-2-13b' in params['model']:
			device = setup_llama2(params['model'], params['gpu_id'])
			for bid, batch in enumerate(batch_prompts):
				print(f"Batch {bid} starts...")
				probs = complete_llama2_13b(device, batch, params['label_dict'], normalize=True)
				for answer in probs:	# probs [bs, class_dim]
					all_raw_answers.append(answer)
		elif 'Llama-2-70b' in params['model']:
			device = setup_llama2(params['model'], params['gpu_id'])
			for bid, batch in enumerate(batch_prompts):
				print(f"Batch {bid} starts...")
				probs = complete_llama2_70b(device, batch, params['label_dict'], normalize=True)
				for answer in probs:	# probs [bs, class_dim]
					all_raw_answers.append(answer)
		else:
			raise NotImplementedError
	return np.asarray(all_raw_answers)


def complete_llama2_13b(device, prompt, label_dict, l=1, normalize=True):
	if isinstance(prompt, str):
		prompt = [prompt]
	input_ids = llama2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
	assert l==1

	total_sequences = llama2_model.generate(input_ids=input_ids['input_ids'].to(device), attention_mask=input_ids['attention_mask'].to(device), max_length=l + len(input_ids['input_ids'][0]), repetition_penalty=1.0, top_k=35, top_p=0.6, temperature=1.0, do_sample=True)
	# we are left padding, so we need to adjust the position IDs
	attention_mask = (total_sequences != 2).float()
	position_ids = attention_mask.long().cumsum(-1) - 1
	position_ids.masked_fill_(attention_mask == 0, 1)
	# get the logits for the context and the next l tokens
	logits = llama2_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()
	# get the top tokens and probs for the generated l tokens
	prediction_probs = torch.softmax(logits[:,-l-1], dim=1).cpu().numpy()
	#bs x 32000
	num_classes = len(label_dict)
	all_test_prediction_probs =[]
	for ind in range(prediction_probs.shape[0]):
		label_probs = [0]*num_classes
		for label_id, label_list in label_dict.items():
			assert len(label_list)==1
			label = label_list[0]
			# llama2 tokenizer split a word into subtokens, we take the average prob of subtokens 
			tokens = llama2_tokenizer.encode(label)[1:]
			label_probs[label_id] = 0
			for tok in tokens:
				# print(tokens, tok, prediction_probs[ind][tok])
				label_probs[label_id] += prediction_probs[ind][tok]
			label_probs[label_id] /= len(tokens)
		if normalize:
			label_probs = [prob/np.sum(label_probs) for prob in label_probs]
		all_test_prediction_probs.append(label_probs)
	return all_test_prediction_probs

def complete_llama2_70b(device, prompt, label_dict, l=1, normalize=True):
	if isinstance(prompt, str):
		prompt = [prompt]
	input_ids = llama2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
	# greedily generate l tokens
	assert l==1
	#查看input_ids的device
	# print(input_ids['input_ids'].device)
	total_sequences = llama2_model.generate(input_ids=input_ids['input_ids'].to(device), attention_mask=input_ids['attention_mask'].to(device), max_length=l + len(input_ids['input_ids'][0]), repetition_penalty=1.0, top_k=35, top_p=0.6, temperature=1.0, do_sample=True)
	# we are left padding, so we need to adjust the position IDs
	attention_mask = (total_sequences != 2).float()
	position_ids = attention_mask.long().cumsum(-1) - 1
	position_ids.masked_fill_(attention_mask == 0, 1)
	# get the logits for the context and the next l tokens
	logits = llama2_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()
	# get the top tokens and probs for the generated l tokens
	prediction_probs = torch.softmax(logits[:,-l-1], dim=1).cpu().numpy()
	#bs x 32000
	num_classes = len(label_dict)
	all_test_prediction_probs =[]
	for ind in range(prediction_probs.shape[0]):
		label_probs = [0]*num_classes
		for label_id, label_list in label_dict.items():
			assert len(label_list)==1
			label = label_list[0]
			# llama2 tokenizer split a word into subtokens, we take the average prob of subtokens 
			tokens = llama2_tokenizer.encode(label)[1:]
			label_probs[label_id] = 0
			for tok in tokens:
				# print(tokens, tok, prediction_probs[ind][tok])
				label_probs[label_id] += prediction_probs[ind][tok]
			label_probs[label_id] /= len(tokens)
		if normalize:
			label_probs = [prob/np.sum(label_probs) for prob in label_probs]
		all_test_prediction_probs.append(label_probs)

	return all_test_prediction_probs


