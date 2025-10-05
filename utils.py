""" We follow existing works to format prompts and generate output probabilities for
	classification datasets including sst5, agnews, dbpedia, rte, and trec
"""
import numpy as np
import time
from copy import deepcopy
import os
import sys
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")
from logging import raiseExceptions

# from huggingface_hub import login
# login(token="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")


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



gpt2_model = None
gpt2_tokenizer = None
llama2_model = None
llama2_tokenizer = None
gemma_model = None
gemma_tokenizer = None


def setup_gpt2(model_name, gpu_id):
	# load the GPT-2 model
	device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
	print('device', device)
	global gpt2_model
	global gpt2_tokenizer
	if gpt2_model is None:
		print(f"Setting up {model_name} model")
		gpt2_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
		gpt2_model.eval().to(device)
		
		gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
		# to batch generation, we pad on the left and mask those positions out.
		gpt2_tokenizer.padding_side = "left"
		gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
		gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
		print("Finished")
	return device


def setup_llama2(model_name, gpu_id):
	# load the Llama-2-13b model
	device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
	print('device', device)
	global llama2_model
	global llama2_tokenizer
	if llama2_model is None:
		print(f"Setting up {model_name} model")
		llama2_model = LlamaForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device) #GPT2LMHeadModel.from_pretrained(model_name)
		llama2_model.eval().to(device)
		llama2_tokenizer = LlamaTokenizer.from_pretrained(model_name)
		# to batch generation, we pad on the left and mask those positions out.
		llama2_tokenizer.padding_side = "left"
		llama2_tokenizer.pad_token = llama2_tokenizer.eos_token
		llama2_model.config.pad_token_id = llama2_model.config.eos_token_id
		print("Finished")
	return device


def setup_gemma(model_name, gpu_id):
	global gemma_model
	global gemma_tokenizer
	dtype = torch.float16 if torch.cuda.is_available() else torch.float32
	device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
	print('device', device)
	if gemma_model is None:
		print(f"Setting up {model_name} model")
		gemma_model = AutoModelForCausalLM.from_pretrained(
			model_name,
			output_hidden_states=True, use_auth_token=True, torch_dtype=dtype).to(device)
		#     device_map={"": device} if torch.cuda.is_available() else None
		# )
		gemma_model.eval().to(device)
		gemma_tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
		gemma_tokenizer.padding_side = "left"
		gemma_tokenizer.pad_token = gemma_tokenizer.eos_token  # Set pad token if missing
		gemma_model.config.pad_token_id = gemma_tokenizer.eos_token_id
		print("Finished")
	return device

# normal
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
		# else:
		# 	assert isinstance(l, str)
		# 	assert params['task_format'] == 'qa'
		# 	l_str = l

		prompt += a_prefix
		prompt += l_str + "\n\n====\n\n"

	prompt += q_prefix
	prompt += infer_sentence + "\n"
	assert a_prefix[-1] == ' '
	prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
	# print("construc prompt", prompt)
	return prompt


# cot
# def construct_prompt(params, priming_sentences, reasoning_sentences, priming_labels, infer_sentence):
# 	"""construct a 1-shot prompt to be fed into the model"""
# 	# take the prompt template and fill in the training and test example
# 	prompt = params["prompt_prefix"]
# 	q_prefix = params["q_prefix"]
# 	r_prefix = params["r_prefix"]
# 	a_prefix = params["a_prefix"]
# 	for s, rea, l in zip(priming_sentences, reasoning_sentences, priming_labels):
# 		prompt += q_prefix
# 		prompt += s + "\n"
# 		if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64):
# 			assert params['task_format'] == 'classification'
# 			l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
# 		# else:
# 		# 	assert isinstance(l, str)
# 		# 	assert params['task_format'] == 'qa'
# 		# 	l_str = l

# 		prompt += a_prefix + r_prefix + rea + " Final answer:" + "\n"
# 		prompt += l_str + "\n\n====\n\n"

# 	prompt += q_prefix
# 	prompt += infer_sentence + "\n"
# 	assert a_prefix[-1] == ' '
# 	prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
	
# 	return prompt

def get_class_probs(params, priming_sentences, priming_labels, infer_sentences, normalize=True):
# def get_class_probs(params, priming_sentences, reasoning_sentences, priming_labels, infer_sentences, normalize=True):
	fix_seed_rerun(params['seed'])
	prompts = []
	for infer_sentence in infer_sentences:
		prompts.append(construct_prompt(params, priming_sentences, priming_labels, infer_sentence))
		# cot
		# prompts.append(construct_prompt(params, priming_sentences, reasoning_sentences, priming_labels, infer_sentence))
	batch_prompts = list(get_batches(prompts, params['bs']))

	all_raw_answers = []
	with torch.no_grad():
		# gpt2
		if 'gpt2' in params['model']:
			device = setup_gpt2(params['model'], params['gpu_id'])
			for bid, batch in enumerate(batch_prompts):
				print(f"Batch {bid} starts...")
				probs = complete_gpt2(device, batch, params['label_dict'], normalize=True)
				for answer in probs:	# probs [bs, class_dim]
					all_raw_answers.append(answer)
		# llama2
		elif 'Llama-2-13b' in params['model']:
			device = setup_llama2(params['model'], params['gpu_id'])
			for bid, batch in enumerate(batch_prompts):
				print(f"Batch {bid} starts...")
				probs = complete_llama2_13b(device, batch, params['label_dict'], normalize=True)
				for answer in probs:	# probs [bs, class_dim]
					all_raw_answers.append(answer)
		elif 'Llama-2-7b' in params['model']:
			device = setup_llama2(params['model'], params['gpu_id'])
			for bid, batch in enumerate(batch_prompts):
				print(f"Batch {bid} starts...")
				probs = complete_llama2_7b(device, batch, params['label_dict'], normalize=True)
				for answer in probs:	# probs [bs, class_dim]
					all_raw_answers.append(answer)
		elif 'gemma-2-2b' in params['model']:
			device = setup_gemma(params['model'], params['gpu_id'])
			for bid, batch in enumerate(batch_prompts):
				print(f"Batch {bid} starts...")
				# print("batch ", batch)
				probs = complete_gemma(device, batch, params['label_dict'], normalize=True)
				for answer in probs:	# probs [bs, class_dim]
					all_raw_answers.append(answer)
		else:
			raise NotImplementedError
	
	return np.asarray(all_raw_answers)

def complete_gpt2(device, prompt, label_dict, l=1, normalize=True):
	if isinstance(prompt, str):
		prompt = [prompt] # the code below assumes a list
	input_ids = gpt2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
	# greedily generate l tokens
	assert l==1
	total_sequences = gpt2_model.generate(input_ids=input_ids['input_ids'].to(device), attention_mask=input_ids['attention_mask'].to(device), max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
   
	# we are left padding, so we need to adjust the position IDs
	attention_mask = (total_sequences != 50256).float()
	position_ids = attention_mask.long().cumsum(-1) - 1
	position_ids.masked_fill_(attention_mask == 0, 1)
	# get the logits for the context and the next l tokens
	logits = gpt2_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()
	# get the top tokens and probs for the generated l tokens
	prediction_probs = torch.softmax(logits[:,-l-1], dim=1).cpu().numpy()
	#bs x 50257
	num_classes = len(label_dict)
	
	all_test_prediction_probs =[]
	for ind in range(prediction_probs.shape[0]):
		label_probs = [0]*num_classes
		for label_id, label_list in label_dict.items():
			assert len(label_list)==1
			label = label_list[0]
			label = " " + label
			token = gpt2_tokenizer.encode(label)[0]
			label_probs[label_id]=prediction_probs[ind][token]
		
		if normalize:
			label_probs = [prob/np.sum(label_probs) for prob in label_probs]
		all_test_prediction_probs.append(label_probs)
	
	return all_test_prediction_probs

def complete_llama2_13b(device, prompt, label_dict, l=1, normalize=True):
	if isinstance(prompt, str):
		prompt = [prompt]
	input_ids = llama2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
	# greedily generate l tokens
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

def complete_llama2_7b(device, prompt, label_dict, l=1, normalize=True):
	if isinstance(prompt, str):
		prompt = [prompt]
	input_ids = llama2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
	# greedily generate l tokens
	assert l==1

	total_sequences = llama2_model.generate(input_ids=input_ids['input_ids'].to(device), attention_mask=input_ids['attention_mask'].to(device), max_length=l + len(input_ids['input_ids'][0]), top_p=0.9, temperature=0.8, do_sample=True)
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

def complete_gemma(device, prompt, label_dict, l=1, normalize=True):
    if isinstance(prompt, str):
        prompt = [prompt]
    # print(prompt)
    # Tokenize inputs
    input_ids = gemma_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    # input_ids = {k: v.to(device) for k, v in input_ids.items()}
    assert l == 1  # Only generating 1 token

    # Generate next token
    total_sequences = gemma_model.generate(
        input_ids=input_ids['input_ids'].to(device),
        attention_mask=input_ids['attention_mask'].to(device),
        max_length=l + len(input_ids['input_ids'][0]),
        # repetition_penalty=1.0,
        # top_k=35,
        # top_p=0.6,
        temperature=1.0,
        do_sample=True,
        pad_token_id=gemma_tokenizer.pad_token_id
    )

    # Compute attention mask and position IDs
    attention_mask = (total_sequences != gemma_tokenizer.pad_token_id).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    # Forward pass to get logits
    logits = gemma_model(
        input_ids=total_sequences,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True
    ).logits.detach().cpu()

    # Get probability distribution over vocabulary for the generated token
    prediction_probs = torch.softmax(logits[:, -l - 1], dim=-1).numpy()

    num_classes = len(label_dict)
    all_test_prediction_probs = []
    for ind in range(prediction_probs.shape[0]):
        label_probs = [0] * num_classes
        for label_id, label_list in label_dict.items():
            assert len(label_list) == 1
            label = label_list[0]
            # Tokenize the label (without special tokens)
            tokens = gemma_tokenizer.encode(label, add_special_tokens=False)
            label_probs[label_id] = 0
            for tok in tokens:
                label_probs[label_id] += prediction_probs[ind][tok]
            label_probs[label_id] /= len(tokens)
        if normalize:
            label_probs = [prob / np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs.append(label_probs)

    return all_test_prediction_probs

# cot
# def complete_gemma(device, params, prompt, label_dict, l=1, normalize=True):
# 	if isinstance(prompt, str):
# 		prompt = [prompt]

# 	print(f"Curr prompt {prompt}")
# 	# Tokenize inputs
# 	input_ids = gemma_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
# 	# input_ids = {k: v.to(device) for k, v in input_ids.items()}
# 	assert l == 1  # Only generating 1 token

# 	# Generate next token
# 	total_sequences = gemma_model.generate(
# 		input_ids=input_ids['input_ids'].to(device),
# 		attention_mask=input_ids['attention_mask'].to(device),
# 		max_length=l + 30 + len(input_ids['input_ids'][0]), # 30 tokens for reasoning (longer will generate hallucinated/nonfactual "====article:...")
# 		# repetition_penalty=1.0,
# 		# top_k=35,
# 		# top_p=0.6,
# 		temperature=1.0,
# 		do_sample=True,
# 		pad_token_id=gemma_tokenizer.pad_token_id
# 	)

# 	outputs = gemma_tokenizer.batch_decode(total_sequences)
# 	for i, out in enumerate(outputs):
# 		print(f"\nCurr batch input {i+1}:\n{out}")

# 	# Compute attention mask and position IDs
# 	attention_mask = (total_sequences != gemma_tokenizer.pad_token_id).float()
# 	position_ids = attention_mask.long().cumsum(-1) - 1
# 	position_ids.masked_fill_(attention_mask == 0, 1)

# 	# Forward pass to get logits
# 	logits = gemma_model(
# 		input_ids=total_sequences,
# 		attention_mask=attention_mask,
# 		position_ids=position_ids,
# 		return_dict=True
# 	).logits.detach().cpu()


# 	label_words = [v[0] for _, v in params["label_dict"].items()]
# 	label_token_map = {
# 		label: gemma_tokenizer.encode(label, add_special_tokens=False)
# 		for label in label_words
# 	}
# 	prediction_probs = []
# 	for i in range(total_sequences.shape[0]):
# 		tokens = total_sequences[i]
# 		# Find last matching label token sequence
# 		found_label = None
# 		found_pos = -1
# 		for label, label_ids in label_token_map.items():
# 			for j in range(len(input_ids['input_ids'][0]), len(tokens) - len(label_ids) + 1):
# 				if torch.equal(tokens[j : j + len(label_ids)], torch.tensor(label_ids, device=tokens.device)):
# 					if j > found_pos:
# 						found_label = label
# 						found_pos = j

# 		if found_label is not None:
# 			target_idx = found_pos + len(label_token_map[found_label]) - 1  # use the last token of the matched label to compute the predicted probability for that label
# 		else:
# 			# Fallback to last non-pad token
# 			target_idx = (tokens != gemma_tokenizer.pad_token_id).nonzero().max().item()

# 		# Get logits and probs
# 		token_logits = logits[i, target_idx, :]
# 		token_probs = torch.softmax(token_logits, dim=-1).numpy()
# 		prediction_probs.append(token_probs)

# 		# # Decode token at target_idx
# 		# predicted_token_id = tokens[target_idx].item()
# 		# decoded_token = gemma_tokenizer.decode([predicted_token_id])
# 		# print(f"found label {found_label} decoded_token {decoded_token}")

# 	prediction_probs = np.array(prediction_probs)


# 	num_classes = len(label_dict)
# 	all_test_prediction_probs = []
# 	for ind in range(prediction_probs.shape[0]):
# 		label_probs = [0] * num_classes
# 		for label_id, label_list in label_dict.items():
# 			assert len(label_list) == 1
# 			label = label_list[0]
# 			# Tokenize the label (without special tokens)
# 			tokens = gemma_tokenizer.encode(label, add_special_tokens=False)
# 			label_probs[label_id] = 0
# 			for tok in tokens:
# 				label_probs[label_id] += prediction_probs[ind][tok]
# 			label_probs[label_id] /= len(tokens)
# 		if normalize:
# 			label_probs = [prob / np.sum(label_probs) for prob in label_probs]
# 		all_test_prediction_probs.append(label_probs)

# 	return all_test_prediction_probs


