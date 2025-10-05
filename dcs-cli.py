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

# ==============================================================================
# 						CORE FUNCTIONS
# ==============================================================================

def fix_seed_rerun(rseed):
	""" Enable reproducibility """
	np.random.seed(rseed)
	random.seed(rseed)
	os.environ['PYTHONHASHSEED'] = str(rseed)


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
		# Should not happen if rule_id is generated correctly
		print('Rule id not supported! Returning the unchanged value...')
		return class_prob


def apply_interpretable_rules(y, sample_pp, w):
	# y[n] is the interpretable rule to be optimized for the class n
	# sample_pp: per-class probabilities for one sample
	sample_pp_new = [fuzzy_rules(sample_pp[n], y[n], w) for n in range(len(sample_pp))]
	# The check for all zeros is a safety feature, but should ideally not be triggered 
	# if rules are correctly designed (e.g., rule_id 17 for no change)
	return sample_pp_new if not all(p == 0 for p in sample_pp_new) else sample_pp

def lblInferWF(y, pp, w):
	# Correct per-class probabilities with interpretable rules
	# y is the rule selection vector
	# pp is the list of per-class probability vectors for all samples
	pred_corrected = [np.argmax(apply_interpretable_rules(y, v, w)) for v in pp]
	return pred_corrected


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
	for j in range(num_class):
		nTrue[j] = B[j] - nError[j]
		rTrue[j] = nTrue[j] / B[j] if B[j] > 0 else 0 # Handle division by zero
		rTrue_j_smoothed = (nTrue[j]+k) / (N+k*num_class*num_class)
		rPred_j_smoothed = (nPred[j]+k) / (N+k*num_class)
		rB_j_smoothed = (B[j]+k) / (N+k*num_class)
		
		# The log is complex and potentially unstable; ensuring the argument is positive
		arg = rTrue_j_smoothed / (rPred_j_smoothed * rB_j_smoothed)
		if arg > 0:
			z3 += -np.log(arg)
		else:
			# Fallback for log(0) or log(<0), should be extremely rare with smoothing 'k'
			z3 += -np.log(1e-9) 

	z2_combs = 0
	for i in range(num_class-1):
		for j in range(i + 1, num_class):
			z2 = z2 + abs(rTrue[i] -  rTrue[j])
			z2_combs +=1
	z2 = z2 / z2_combs if z2_combs > 0 else 0 # Handle division by zero

	z = alpha * z1 + beta * N*z2 + tau * N*z3
	return z


def compute_bias(class_acc):
	num_classes = len(class_acc)
	# Check for zero classes where accuracy might be NaN or zero
	valid_acc = [acc for acc in class_acc if not np.isnan(acc)]
	
	if len(valid_acc) < 2:
		return 0.0 # Cannot compute bias with less than 2 classes

	bias = sum(abs(valid_acc[i] - valid_acc[j]) for i in range(len(valid_acc)-1) for j in range(i + 1, len(valid_acc)))
	
	num_combinations = len(list(combinations(range(len(valid_acc)), 2)))
	bias /= num_combinations if num_combinations > 0 else 1
	return bias

# ==============================================================================
# 					FIND BEST RULES
# ==============================================================================

def find_optimal_correction_rules(config, pp_raw, lbl_raw, pred_raw, num_class):
	"""
	Internalizes the hyperparameter search and returns the optimal 
	correction vector (y_best) and weight scale (w).
	
	Args:
		config (Config): The configuration object.
		pp_raw (list): Raw per-class probabilities.
		lbl_raw (list): Raw true labels.
		pred_raw (list): Raw predictions (not used, but kept for completeness).
		num_class (int): Number of classes.

	Returns:
		tuple: (y_best_final, w_final), the optimal rule vector and weight scale.
	"""
	
	rseed = 1 # Fixed seed for reproducibility
	fix_seed_rerun(rseed)
	
	# 1. Hyperparameter Space Definition
	k = 4000
	alphas = [0, 0.5, 1, 2]
	betas = [0.5, 1, 2.7, 3, 10]
	taus = [0, 0.2, 1, 5]
	num_fuzzy = [19] 
	num_ws = [30, 50, 70, 90]
	
	# Filtering criteria for hyperparameters (from your original code)
	def filter_params(alpha, beta, tau):
		if alpha == 0 and beta == 0 and tau == 0: return False
		if alpha == 0 and beta == 0 and tau != 1: return False
		if alpha == 0 and beta != 1 and tau == 0: return False
		if alpha != 1 and beta == 0 and tau == 0: return False
		return True

	# 2. Split into Opt and Dev Sets (0.95/0.05)
	np.random.seed(rseed)
	N_raw = len(lbl_raw)
	# Create indices for the 95% Opt set and 5% Dev set
	opt_inds = np.random.choice(N_raw, size=int(0.95 * N_raw), replace=False)
	dev_inds = [i for i in range(N_raw) if i not in opt_inds]

	pp_opt = [pp_raw[i] for i in opt_inds]
	lbl_opt = [lbl_raw[i] for i in opt_inds]
	
	pp_dev = [pp_raw[i] for i in dev_inds]
	lbl_dev = [lbl_raw[i] for i in dev_inds]
	
	B_dict_opt = Counter(lbl_opt)
	B_opt = [B_dict_opt[x] if x in B_dict_opt else 0 for x in range(num_class)]
	
	# Load SA hyperparameters from config (or use internal defaults)
	T_min = getattr(config, 'T_min', 1e-4)
	r_temp = getattr(config, 'r_temp', 0.5)
	iter_min = getattr(config, 'iter_min', num_class * 3) # Empirically: 3x num of vars
	iter_max = getattr(config, 'iter_max', 1000)
	n_out_loop = getattr(config, 'n_out_loop', 1000)
	n_in_loop = getattr(config, 'n_in_loop', 50)
	low_tem = getattr(config, 'low_tem', 0.99)
	
	best_dev_acc = -1.0
	best_dev_bias = np.inf
	y_best_final = None
	w_final = None

	# 3. Hyperparameter Search Loop
	for alpha in alphas:
		for beta in betas:
			for tau in taus:
				if not filter_params(alpha, beta, tau):
					continue
					
				for num_f in num_fuzzy:
					for num_w in num_ws:
						# Define weight scale and number of rules for current iteration
						w = [(i + 1) / num_w for i in range(num_w)]
						num_r = num_f + num_w
						default_rule_id = num_r - 1
						
						# --- Simulated Annealing (SA) Process ---
						
						# Initialize SA parameters
						y = [default_rule_id for _ in range(num_class)]
						y_best = y.copy()
						y_cur = y.copy()
						nTrue = [0] * num_class
						rTrue = [0] * num_class
						
						pred = lblInferWF(y_best, pp_opt, w)
						z_cur = ObjFunction(pred, lbl_opt, B_opt, nTrue, rTrue, alpha, beta, tau, k, num_class)
						z_best = z_cur
						current_temp = r_temp # Reset temperature for new hyperparam set

						for T in range(n_out_loop):
							n_generate = 0
							n_accept = 0
							for mk in range(n_in_loop):
								y_new = y_cur.copy()
								ii = random.randint(0, num_class - 1)
								jj = random.randint(0, num_r - 1)
								while jj == y_cur[ii]: # Note: checking against y_cur, not initial y
									jj = random.randint(0, num_r - 1)
								
								y_new[ii] = jj
								pred_corrected = lblInferWF(y_new, pp_opt, w)
								z_new = ObjFunction(pred_corrected, lbl_opt, B_opt, nTrue, rTrue, alpha, beta, tau, k, num_class)
								n_generate += 1
								
								if z_new <= z_cur:
									n_accept += 1
									y_cur = y_new.copy()
									z_cur = z_new
									if z_new < z_best:
										z_best = z_new
										y_best = y_new.copy() # FOUND NEW SA BEST
								elif random.uniform(0, 1) < np.exp((z_cur - z_new) / current_temp):
									y_cur = y_new.copy()
									z_cur = z_new
								
								if n_accept >= iter_min or n_generate >= iter_max:
									break
							
							current_temp *= low_tem
							if current_temp < T_min:
								break
						
						# --- Dev Set Evaluation (Selection Criterion) ---
						
						# Apply the SA-optimized rules (y_best) to the DEV set
						pred_dev_corrected = lblInferWF(y_best, pp_dev, w)
						dev_acc = accuracy_score(lbl_dev, pred_dev_corrected)
						
						dev_matrix = confusion_matrix(lbl_dev, pred_dev_corrected, labels=range(num_class))
						# Check if any class has zero true samples, leading to division by zero
						sum_axis_1 = dev_matrix.sum(axis=1)
						with np.errstate(divide='ignore', invalid='ignore'):
							dev_class_acc = np.where(sum_axis_1 != 0, dev_matrix.diagonal() / sum_axis_1, np.nan)
							
						dev_bias = compute_bias(dev_class_acc)

						# 4. Selection Logic (Highest Acc, then Lowest Bias)
						is_new_best = False
						if dev_acc > best_dev_acc:
							is_new_best = True
						elif dev_acc == best_dev_acc and dev_bias < best_dev_bias:
							is_new_best = True

						if is_new_best:
							best_dev_acc = dev_acc
							best_dev_bias = dev_bias
							y_best_final = y_best
							w_final = w
							print(f"**NEW BEST** | A:{alpha} B:{beta} T:{tau} Ws:{num_w} | Dev Acc: {best_dev_acc:.4f} | Dev Bias: {best_dev_bias:.4f} | y_best: {y_best_final}")
						
	
	if y_best_final is None:
		# Fallback: if no valid combination was found, return a no-op rule (rule_id 17 for 'No Change')
		default_rule = num_f + num_ws[0] - 1 # Assuming num_f is 19 and num_ws[0] is 30, this is 19+30-1=48. No-change is 17. 
		# We must ensure the default rule is in the valid range for the final num_r
		# Let's use a standard no-change where possible, or fall back to the largest rule set's no-op index.
		# Since num_r changes, we must be careful. Let's return a simple all-17 rule.
		print("Warning: No optimal rules found. Falling back to 'No Change' rule set.")
		max_num_r = max(num_f) + max(num_ws)
		default_y = [17] * num_class 
		default_w = [(i + 1) / max(num_ws) for i in range(max(num_ws))] # Use max Ws for default w
		return default_y, default_w

	return y_best_final, w_final

# ==============================================================================
# 					FINAL DEPLOYMENT & CLI USAGE
# ==============================================================================

def load_data(vec_dir):
	"""Loads raw data from opt.txt and test.txt files."""
	file_path = os.path.join(vec_dir, 'opt.txt')
	
	with open(file_path, 'r') as f:
		line = f.readline()
	
	parts = line.strip().split()
	if len(parts) < 3:
		raise ValueError(f"File {file_path} is malformed or empty.")
	num_class = len(parts[3:])

	lbl_raw, pp_raw, pred_raw = [], [], []
	with open(file_path, 'r') as file:
		for line in file:
			parts = line.strip().split()
			cur_lbl = int(parts[1])
			cur_pred = int(parts[2])
			p_vec = [float(parts[n + 3]) for n in range(num_class)]
			
			lbl_raw.append(cur_lbl) 
			pred_raw.append(cur_pred) 
			pp_raw.append(p_vec)

	# Load test data for final evaluation
	file_path_test = os.path.join(vec_dir, 'test.txt')
	raw_test = open(file_path_test).readlines()
	lbl_test = [int(x.strip().split()[1]) for x in raw_test]
	pp_test = [[float(x.strip().split()[n + 3]) for n in range(num_class)] for x in raw_test]

	return lbl_raw, pp_raw, pred_raw, num_class, lbl_test, pp_test


def deployment_cli_main(config):
	"""Main function for the deployment/CLI module."""
	
	print(f"Loading data from: {config.vec_dir}")
	lbl_raw, pp_raw, pred_raw, num_class, lbl_test, pp_test = load_data(config.vec_dir)
	print(f"Total samples loaded: {len(lbl_raw)} (Opt) and {len(lbl_test)} (Test)")
	print(f"Number of classes: {num_class}")

	start_time = time.time()
	
	# === CORE: Find the Optimal Correction Rules ===
	optimal_y, optimal_w = find_optimal_correction_rules(
		config, 
		pp_raw, 
		lbl_raw, 
		pred_raw, 
		num_class
	)
	
	end_tuning_time = time.time()
	print("\n" + "="*50)
	print("HYPERPARAMETER TUNING COMPLETE")
	print(f"Tuning Time: {end_tuning_time - start_time:.2f} seconds")
	print(f"Final Selected Correction Indices (y_best): {optimal_y}")
	print(f"Final Weight Scale (w) Size: {len(optimal_w)}")
	print("="*50)
	
	# === FINAL OUTPUT: Test Set Evaluation using Optimal Rules ===
	
	if optimal_y is not None:
		print("\n--- Applying Optimal Rules to Test Set ---")
		
		# 1. Correct test predictions
		pred_test_corrected = lblInferWF(optimal_y, pp_test, optimal_w)
		
		# 2. Evaluate
		test_acc = accuracy_score(lbl_test, pred_test_corrected)
		matrix = confusion_matrix(lbl_test, pred_test_corrected, labels=range(num_class))
		
		sum_axis_1 = matrix.sum(axis=1)
		with np.errstate(divide='ignore', invalid='ignore'):
			test_class_acc = np.where(sum_axis_1 != 0, matrix.diagonal() / sum_axis_1, np.nan)
			
		test_bias = compute_bias(test_class_acc)

		print(f'Test Accuracy: {test_acc:.4f}')
		print(f'Test Class Accuracies: {test_class_acc}')
		print(f'Test Bias: {test_bias:.4f}')
		
		# Optional: Full classification report
		# print("\nClassification Report:\n", classification_report(lbl_test, pred_test_corrected))

	total_run_time = time.time() - start_time
	print(f"\nTotal Program Execution Time: {total_run_time:.2f} seconds")
	
	# In a real deployment scenario, you would save `optimal_y` and `optimal_w`
	# to a file (e.g., a pickle or JSON) for later use by the prediction module.
	return optimal_y, optimal_w


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-d', "--vec_dir", default=None, help="Vector directory to raw output class proabilities by an LLM.")
	parser.add_argument('-c', '--config', type=str, default=None)

	args = parser.parse_args()
	if args.config is not None:
		cur_config_path = args.config
	else:
		cur_config_path = os.path.join("config", "default_params.json")

	update_config = vars(args)
	config = Config(cur_config_path, update_config)
	
	# The deployment CLI now calls the main logic function
	# which handles the full hyperparameter search and selection.
	optimal_indices, optimal_weights = deployment_cli_main(config)
	
	# Example of the final desired output:
	# e.g., print(f"Selected Correction Indices: {optimal_indices}")