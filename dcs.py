"""
Ensemble Debiasing Across Class and Sample Levels for Fairer Prompting Accuracy
Usage: 
	nohup python dcs.py \
		-c config/default_params.json \
		--vec_dir vectors/llama2-13b/ddi_llama2-13b_shot1_seed1 \
		> res-ddi-seed1.log 2>&1 &
"""

import argparse
import math
import random
import time
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import combinations

from config import Config
from vectors.dataset_utils import sample_subset_by_lbl


def fix_seed_rerun(rseed):
	""" Enable reproducibility """
	np.random.seed(rseed)
	random.seed(rseed)
	os.environ['PYTHONHASHSEED'] = str(rseed)


def lblInferWF(y, pp, w):
	# Correct per-class probabilities with interpretable rules
	# y is the rule selection vector
	pred_corrected = [np.argmax(apply_interpretable_rules(y, v, w)) for v in pp]
	return pred_corrected


def apply_interpretable_rules(y, sample_pp, w):
	# y[n] is the interpretable rule to be optimized for the class n
	sample_pp_new = [fuzzy_rules(sample_pp[n], y[n], w) for n in range(len(sample_pp))]
	return sample_pp_new if sample_pp_new != [0]*len(sample_pp) else sample_pp


def fuzzy_rules(class_prob, rule_id, w):
	# slope 2
	if rule_id==0:
		return 0 if class_prob>0.5 else 1.0-2.0*class_prob
	elif rule_id==1:
		if class_prob<0.5:
			return 2.0*class_prob
		return 2.0-2.0*class_prob
	elif rule_id==2:
		return 0 if class_prob<0.5 else 2.0*class_prob-1.0
	# slope 4
	elif rule_id==3:
		return 0 if class_prob>0.25 else 1.0-4.0*class_prob
	elif rule_id==4:
		if class_prob>0.5:
			return 0
		if class_prob<=0.25:
			return 4.0*class_prob
		return 2.0-4.0*class_prob
	elif rule_id==5:
		if class_prob>0.75 or class_prob<0.25:
			return 0
		if class_prob<=0.5:
			return 4.0*class_prob-1.0
		return 3.0-4.0*class_prob
	elif rule_id==6:
		if class_prob<0.5:
			return 0
		if class_prob<=0.75:
			return 4.0*class_prob-2.0
		return 4.0-4.0*class_prob
	elif rule_id==7:
		return 0 if class_prob<0.75 else 4.0*class_prob-3.0
	# slope 8
	elif rule_id==8:
		if class_prob>0.125:
			return 0
		return 1.0-8.0*class_prob
	elif rule_id==9:
		if class_prob>0.25:
			return 0
		if class_prob<=0.125:
			return 8.0*class_prob
		return 2.0-8.0*class_prob
	elif rule_id==10:
		if class_prob>0.375 or class_prob<0.125:
			return 0
		elif class_prob<=0.25:
			return 8.0*class_prob-1.0
		return 3.0-8.0*class_prob
	elif rule_id==11:
		if class_prob>0.5 or class_prob<0.25:
			return 0
		elif class_prob<=0.375:
			return 8.0*class_prob-2.0
		return 4.0-8.0*class_prob
	elif rule_id==12:
		if class_prob>0.625 or class_prob<0.375:
			return 0
		elif class_prob<=0.5:
			return 8.0*class_prob-3.0
		return 5.0-8.0*class_prob
	elif rule_id==13:
		if class_prob>0.75 or class_prob<0.5:
			return 0
		elif class_prob<=0.625:
			return 8.0*class_prob-4.0
		return 6.0-8.0*class_prob
	elif rule_id==14:
		if class_prob>0.875 or class_prob<0.625:
			return 0
		elif class_prob<=0.75:
			return 8.0*class_prob-5.0
		return 7.0-8.0*class_prob
	elif rule_id==15:
		if class_prob<0.75:
			return 0
		elif class_prob<=0.875:
			return 8.0*class_prob-6.0
		return 8.0-8.0*class_prob
	elif rule_id==16:
		if class_prob<0.875:
			return 0
		return 8.0*class_prob-7.0
	elif rule_id==17:
		# No Change
		return class_prob
	elif rule_id==18:
		return 1.0-class_prob
	elif rule_id > 18:
		# Weight correction
		return class_prob*w[rule_id-19]
	else:
		print('Rule id not supported! Returning the unchanged value...')
		return class_prob


def ObjFunction(pred_calib, lbl, B, nTrue, rTrue, alpha, beta, tau, k, num_class):
	N = len(lbl)
	z1 = 0
	z2 = 0
	z3 = 0
	nError = [0] * num_class 
	nPred = [0] * num_class
	for m in range(N):
		nPred[pred_calib[m]] += 1
		if lbl[m] != pred_calib[m]:
			nError[lbl[m]] = nError[lbl[m]] + 1
			z1 += 1
			# if lbl[m] != 0:
			#     z1 += 1
	for j in range(num_class):
		nTrue[j] = B[j] - nError[j]
		rTrue[j] = nTrue[j] / B[j]
		rTrue_j_smoothed = (nTrue[j]+k) / (N+k*num_class*num_class)
		rPred_j_smoothed = (nPred[j]+k) / (N+k*num_class)
		rB_j_smoothed = (B[j]+k) / (N+k*num_class)
		z3 += -np.log(rTrue_j_smoothed / (rPred_j_smoothed*rB_j_smoothed))
	z2_combs = 0
	for i in range(num_class-1):
		for j in range(i + 1, num_class):
			z2 = z2 + abs(rTrue[i] -  rTrue[j])
			z2_combs +=1
	z2 /= z2_combs
	z = alpha * z1 + beta * N*z2 + tau * N*z3
	return z


def compute_bias(class_acc):
	num_classes = len(class_acc)
	bias = sum(abs(class_acc[i] - class_acc[j]) for i in range(num_classes-1) for j in range(i + 1, num_classes))
	bias /= len(list(combinations(range(num_classes), 2)))
	return bias


def save_to_file(out_fp, res, mode='a'):
	with open(out_fp, mode) as f:
		f.write(res + '\n')

# Main
def main(config):
	# You may specify your own hyperparameters in config
	# k = config.k
	# beta = config.beta
	# tau = config.beta
	# num_r = config.num_r
	# rseed = config.rseed

	# Hyperparameter selection
	k = 4000
	alphas = [0, 0.5, 1, 2]
	betas = [0.5, 1, 2.7,3,10]
	taus = [0, 0.2, 1, 5]
	num_fuzzy =  [19] # Sample-level correction: number of fuzzy triangular membership functions (MFs), 2 triangular MF for slope (+/-)1, 3 for slope (+/-)2, 5 for slope (+/-)4, 9 for slope (+/-)8
	num_ws = [30, 50, 70, 90] # Class-level correction: eight scale
	rseed = 1
	# You may reduce opt set size
	# (Faster with comparable results on some datasets)
	num_samples = ['full'] # [10, 50, 100, 500, 1000]

	for num_sample in num_samples:
		# Read in raw data
		lbl_raw = []
		# Predicted per-class probabilities (N-dim)
		pp_raw = []
		# Predictions
		pred_raw = []
		# Read labels and predicted token likelihoods
		file_path = os.path.join(config.vec_dir, 'opt.txt')
		ds = config.vec_dir.split('/')[-1].split('_')[0]
		print('==ds==', ds)
		if ds == 'pubmedqa' and num_sample == 1000:
			break

		with open(file_path, 'r') as f:
			line = f.readline()
		num_class = len(line.strip().split()[3:])
		print('num_class ', num_class)

		with open(file_path, 'r') as file:
			for line in file:
				parts = line.strip().split()
				cur_lbl = int(parts[1])
				lbl_raw.append(cur_lbl) 
				cur_pred = int(parts[2])
				pred_raw.append(cur_pred) 
				p_vec = [float(parts[n + 3]) for n in range(num_class)]
				pp_raw.append(p_vec)

		# Split raw opt set into opt and dev by 0.95/0.05
		np.random.seed(rseed)
		opt_inds = np.random.choice(len(lbl_raw), size=int(0.95*len(lbl_raw)), replace=False)
		pp = [pp_raw[i] for i in opt_inds]
		lbl = [lbl_raw[i] for i in opt_inds]
		preds_not_used = [pred_raw[i] for i in opt_inds]

		dev_inds = [x for x in range(len(lbl_raw)) if x not in opt_inds]
		pp_dev = [pp_raw[i] for i in dev_inds]
		lbl_dev = [lbl_raw[i] for i in dev_inds]
		B_dict_dev = Counter(lbl_dev)
		B_dev = [B_dict_dev[x] for x in sorted(B_dict_dev.keys())]
		# print('Dev support:', B_dev)
		print(type(num_sample))
		if num_sample == 'full':
			print(f"===running on full opt set, {len(lbl)} opt samples===")
		elif num_sample > len(pp):
			print(f"==={num_sample} is greater than the size of the input dataset ({len(pp)})! Skipping...")
			break
		else:
			if num_sample == 10 and ds == 'dbpedia':
				num_sample = 15 # dbpedia: 14 classes
			pp, lbl, _ = sample_subset_by_lbl(config.vec_dir, num_class, pp, lbl, preds_not_used, num_sample, rseed)
			print(f"===Using {config.vec_dir} {len(pp)} opt samples===")

		B_dict = Counter(lbl)
		B = [B_dict[x] if x in B_dict else 0 for x in range(num_class)]
		print('Opt set support:', B)
		
		for alpha in alphas:
			for beta in betas:
				for tau in taus:
					if alpha == 0 and beta == 0 and tau == 0:
						continue
					elif alpha == 0 and beta == 0 and tau != 1:
						continue
					elif alpha == 0 and beta != 1 and tau == 0:
						continue
					elif alpha != 1 and beta == 0 and tau == 0:
						continue
					else:
						print('===exp starts===')
					for num_f in num_fuzzy:
						for num_w in num_ws:						
							fix_seed_rerun(rseed)
							start_time = time.time()

							# Weight scale
							w = [(i+1) / num_w for i in range(num_w)]

							# Randomly initialize a rule selection vector y for output classes
							num_r = num_f + num_w
							default_rule_id = num_r - 1
							y = [default_rule_id for _ in range(num_class)]

							# Select hyperparameters based on acc. on a dev set
							print('===opt_size: {} r: {} alpha: {} beta: {} tau: {} starts==='.format(num_sample, num_r, alpha, beta, tau))


							# Load SA hyperparameters from config
							T_min = config.T_min
							r_temp = config.r_temp
							iter_min = config.iter_min
							iter_max = config.iter_max
							n_out_loop = config.n_out_loop
							n_in_loop = config.n_in_loop
							low_tem = config.low_tem

							nTrue = [0] * num_class
							rTrue = [0] * num_class
							header_format = "{:^10} {:^8} {:^10} {:^10} {:^8} {:^8} {:^12} {:^12} {:^12} {:^12} "
							header = header_format.format("Iter.", "Temp", "Accept rate", "Accept sol.", "Num of sol.",  "Avg. z","Min z", "Max z","Total run time","Iter duration")
							print(header)

							# Run simulated annealing
							# Rule vector y contains the rule id for all classes, e.g., [4, 4, 4, 4, 4]
							y_best = y.copy()
							y_cur = y.copy()
							pred = lblInferWF(y_best, pp, w)
							z_cur = ObjFunction(pred, lbl, B, nTrue, rTrue, alpha, beta, tau, k, num_class)
							z_best = z_cur
							for T in range(n_out_loop):
								start_iteration_time = time.time()
								z_total = 0
								z_max = -np.inf
								z_min =  np.inf
								n_generate=0
								n_accept=0
								for mk in range(n_in_loop):
									y_new = y_cur.copy()
									# Start by randomly selecting a to-be-corrected class, denoted as ii
									ii = random.randint(0, num_class-1)
									# jj is a randomly initialized rule id, in the range of num_r
									# core idea of SA: replace class ii's rule id y[ii] by jj and check if objective is improved, iterately
									jj = random.randint(0, num_r-1)
									# Make sure jj is not the same as the current selected rule id for the ii class
									while jj == y[ii]:
										jj = random.randint(0, num_r-1)
									y_new[ii] = jj
									pred_corrected = lblInferWF(y_new, pp, w)
									z_new = ObjFunction(pred_corrected, lbl, B, nTrue, rTrue, alpha, beta, tau, k, num_class)
									n_generate+=1
									z_total += z_new
									# record max and min z during SA
									z_min = min(z_min, z_new)
									z_max = max(z_max, z_new)
									# Update the optimal solution
									# SA allows worse z to jump out local minima, help avoid local minima
									# z_cur keeps record of current result (could be worse than z_best)
									# z_best keeps record of historial best result
									if z_new <= z_cur:
										n_accept += 1
										y_cur = y_new.copy()
										z_cur = z_new
										if z_new < z_best:
											z_best = z_new
											y_best = y_new.copy()
									elif random.uniform(0, 1) < np.exp((z_cur - z_new) / r_temp):
										y_cur = y_new.copy()
										z_cur = z_new
									# SA inner loop stopping criteria
									# For text classification
									# If n_accept>=iter_min or n_generate>=iter_max:
									# For custom classification tasks on vision and more, early stop iter_min needs to be larger
									# Empirically, 3x num of vars
									if n_accept>=iter_min or n_generate>=iter_max:
										break
								r_temp = r_temp * low_tem
								end_iteration_time = time.time()
								iteration_duration = end_iteration_time - start_iteration_time
								accept_rate = n_accept / n_generate if n_generate > 0 else 0
								total_run_time = end_iteration_time - start_time
								z_average = z_total / n_generate
								iteration_info_format = "{:^10d} {:^15.3f} {:^10.4f} {:^15d} {:^15d} {:^16d} {:^15d} {:^20d} {:^17.2f} {:^17.2f}"
								iteration_info = iteration_info_format.format(T, r_temp, accept_rate, n_accept, n_generate, int(z_average),
																			  int(z_min), int(z_max), total_run_time, iteration_duration)
								print(iteration_info)
								# SA outer loop stopping criterion
								if r_temp < T_min:
									break

							# Update predictions with optimal rules
							pred_corrected = lblInferWF(y_best, pp, w)
							z_check=ObjFunction(pred_corrected, lbl, B, nTrue, rTrue, alpha, beta, tau, k, num_class)
							print(z_check, z_best)
							if z_check != z_best:
								print('z_check != z_best')
								print('____________________')
							print(f'Number of fuzzy+weight rules: {num_r}')
							print('The selected interpretable rule ids: '+str(y_best))
							print('Objective function value:  '+str(z_best))

							# Opt set evaluation
							opt_acc = accuracy_score(lbl, pred_corrected)
							print('===Opt set acc===', opt_acc)
							score_report = classification_report(lbl, pred_corrected)
							print(score_report)
							opt_matrix = confusion_matrix(lbl, pred_corrected)
							opt_class_acc = opt_matrix.diagonal()/opt_matrix.sum(axis=1)
							print('opt class acc.', opt_class_acc)
							opt_bias = compute_bias(opt_class_acc)
							print('opt bias ', opt_bias)
							# opt_pmi_total, opt_pmi_class = compute_pmi(opt_class_acc, pred_corrected, B, len(lbl), k)
							# print('opt PMI ', opt_pmi_total, opt_pmi_class)

							# Dev evaluation
							pred_dev_corrected = lblInferWF(y_best, pp_dev, w)
							dev_acc = accuracy_score(lbl_dev, pred_dev_corrected)
							print('===dev acc===', dev_acc)
							score_report = classification_report(lbl_dev, pred_dev_corrected)
							print(score_report)
							dev_matrix = confusion_matrix(lbl_dev, pred_dev_corrected)
							dev_class_acc = dev_matrix.diagonal()/dev_matrix.sum(axis=1)
							print('dev class acc.', dev_class_acc)
							dev_bias = compute_bias(dev_class_acc)
							print('dev bias ', dev_bias)
							# dev_pmi_total, dev_pmi_class = compute_pmi(dev_class_acc, pred_dev_corrected, B_dev, len(lbl_dev), k)
							# print('dev PMI ', dev_pmi_total, dev_pmi_class)

							end_time = time.time()
							run_time = end_time - start_time
							print("Program execution time:"+str("{:.4}".format(run_time))+'  seconds')

							# Get the best y_best on dev set
							pred_test, lbl_test = [], []
							# Read labels and predicted token likelihoods
							file_path = os.path.join(config.vec_dir, 'test.txt')
							raw_test = open(file_path).readlines()
							lbl_test = [int(x.strip().split()[1]) for x in raw_test]
							pp_test = [[float(x.strip().split()[n + 3]) for n in range(num_class)] for x in raw_test]

							assert len(pp_test) == len(lbl_test)

							# Correct test predictions
							B_dict = Counter(lbl_test)
							B_test = [B_dict[x] for x in sorted(B_dict.keys())]
							# print('Test support:', B_test)

							pred_test_corrected = lblInferWF(y_best, pp_test, w)
							test_acc = accuracy_score(lbl_test, pred_test_corrected)
							print('===Test acc===', test_acc)
							score_report = classification_report(lbl_test, pred_test_corrected)
							print(score_report)
							matrix = confusion_matrix(lbl_test, pred_test_corrected)
							test_class_acc = matrix.diagonal()/matrix.sum(axis=1)
							print('test class acc. ', test_class_acc)
							test_bias = compute_bias(test_class_acc)
							print('test bias ', test_bias)
							# test_pmi_total, test_pmi_class = compute_pmi(test_class_acc, pred_test_corrected, B_test, len(lbl_test), k)
							# print('test PMI ', test_pmi_total, test_pmi_class)
							print('=======opt_size {} rseed {} alpha {} beta {} tau {} r {} ends========='.format(num_sample, rseed, alpha, beta, tau, num_r))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-d', "--vec_dir", default=None, help="Vector directory to raw output class proabilities by an LLM.")
	# Load config
	parser.add_argument('-c', '--config', type=str, default=None)

	args = parser.parse_args()
	if args.config is not None:
		cur_config_path = args.config
	else:
		cur_config_path = os.path.join("config", "default_params.json")

	update_config = vars(args)
	print('Update config', update_config)
	config = Config(cur_config_path, update_config)
	main(config)



