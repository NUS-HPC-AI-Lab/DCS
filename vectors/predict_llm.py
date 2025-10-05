import argparse
from dataset_utils import read_dataset, sample_subset, sample_subset_by_lbl, preproc_input
from utils import *
import random
import json
from copy import deepcopy
import numpy as np


def main(model, dataset, num_shots, num_seeds, start_seed, bs, split, gpu_id):
	param_dict = {'model': model, 'dataset': dataset, 'bs': bs, 'split': split, 'gpu_id':gpu_id}
	param_combs = []  
	for num_shot in num_shots:
		for seed in range(start_seed, start_seed+num_seeds):                 
			p = deepcopy(param_dict)
			p['seed'] = seed
			p['num_shots'] = num_shot
			p['expr_name'] = f"{p['dataset']}_{p['split']}_{p['model']}_shot{p['num_shots']}_seed{p['seed']}"
			param_combs.append(p)
	# print('===param_combs===', param_combs)
	save_output_class_probs(param_combs)
	
def save_output_class_probs(params_list):
	"""
	Get prediction probabilities over classes
	"""
	for _, params in enumerate(params_list):
		print("===Current run=== ", params['expr_name'])
		# Load data and sample few-shot priming examples
		all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = read_dataset(params)
		print(f'All train/test samples: {len(all_train_labels)}, {len(all_test_labels)}')
		num_class = len(list(set(all_train_labels)))
		print(f'Num class: {num_class}')

		if params['num_shots'] == num_class:
			priming_sentences, priming_labels = sample_subset_by_lbl(all_train_sentences, all_train_labels, params['num_shots'], None, rseed=params['seed'])
		else:	
			priming_sentences, priming_labels = sample_subset(all_train_sentences, all_train_labels, params['num_shots'], None, rseed=params['seed'])

		# Preprocess inference data
		if params['split'] == 'opt':
			infer_sentences, infer_labels = all_train_sentences, all_train_labels
		elif params['split'] == 'test':
			infer_sentences, infer_labels = all_test_sentences, all_test_labels
			if params['dataset'] in ['agnews']:
					# Keep first 5k test sentences for test evaluations
					infer_sentences = infer_sentences[:5000]
					infer_labels = infer_labels[:5000]
			elif params['dataset'] in ['dbpedia']:
				np.random.seed(0)
				selected_inds = np.random.choice(len(infer_labels), size=5000, replace=False)
				infer_sentences = [infer_sentences[i] for i in selected_inds]
				infer_labels = [infer_labels[i] for i in selected_inds]

		# optional: save inference data
		# if not os.path.exists(os.path.join(out_dir, out_fp)):
		# 	with open(os.path.join(out_dir, out_fp), 'w') as f:
		# 		for s, l in zip(infer_sentences, infer_labels):
		# 			# print(s, l )
		# 			out_dict = {'TEXT': s, 'LBL': l}
		# 			f.write(json.dumps(out_dict) + '\n')

		
		print(f"Start getting output class probs for {len(infer_labels)} inferences...")

		# Obtain model's output per-class probabilities on optimization/test examples
		all_class_probs = get_class_probs(params, priming_sentences, priming_labels, infer_sentences)
		assert len(all_class_probs) == len(infer_labels)

		if 'Llama-2-13b' in params['model']:
			model_n = 'llama2-13b'
		elif 'Llama-2-70b' in params['model']:
			model_n = 'llama2-70b'
		else:
			model_n = params['model']
		vec_dir = f"vecs_{model_n}"
		if not os.path.exists(vec_dir):
			os.makedirs(vec_dir)
		res_dir = os.path.join(vec_dir, f"{params['dataset']}_{model_n}_shot{params['num_shots']}_seed{params['seed']}")
		if not os.path.exists(res_dir):
			os.makedirs(res_dir)
		print(f"Saving output class probabilities to {res_dir}...")
		save_vecs_txt(res_dir, all_class_probs, infer_labels, params['split'])

def save_vecs_txt(res_dir, predicted_class_probs, labels, split):
	# Get argmax predictions
	preds = []
	for class_prob in predicted_class_probs:
		pred = np.argmax(class_prob)
		preds.append(pred)

	with open(os.path.join(res_dir, f"{split}.txt"), 'w') as f:
		# f.write('ID\tLABEL\tPRED\tPROB\n')
		for i in range(len(labels)):
			class_prob_i = ' '.join([format(x, '.6f') for x in predicted_class_probs[i]])
			f.write(str(i)+' '+str(labels[i])+' '+str(preds[i])+' '+str(class_prob_i)+'\n')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', dest='model', action='store', required=True, help='Path to model.')
	parser.add_argument('--dataset', dest='dataset', action='store', required=True, help='Name of dataset, e.g., ddi.')
	parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='Number of seeds(runs) for the optimization set.', type=int)
	parser.add_argument('--start_seed', dest='start_seed', action='store', required=True, help='Start seed(run) for the optimization set.', type=int)
	parser.add_argument('--num_shots', dest='num_shots', action='store', required=True, help='Number of few-shot demonstrations.')
	parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
						help='Batch size.')
	parser.add_argument("--split", required=True, type=str, default="random", choices=["opt", "test"], help="Dataset split.")
	parser.add_argument("--gpu_id", required=True, type=int, default="0", help="GPU ID to run predictions.")

	args = parser.parse_args()
	args = vars(args)
	args['num_shots'] = [int(s.strip()) for s in args['num_shots'].split(",")]

	main(**args)


