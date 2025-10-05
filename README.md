# Ensemble Debiasing Across Class and Sample Levels for Fairer Prompting Accuracy [COLM 2025]

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2503.05157-brightgreen.svg)](https://arxiv.org/abs/2503.05157)
[![COLM 2025](https://img.shields.io/badge/COLM-2025-violet.svg)](https://openreview.net/forum?id=63c7hTrUCh)
[![Huggingface](https://img.shields.io/badge/🤗-HF-yellow)](https://huggingface.co/papers/2503.05157)

</div>

## 👥 Authors

**Ruixi Lin**<sup>1</sup>, **Ziqiao Wang**<sup>1</sup>, **Yang You**<sup>1</sup>

<sup>1</sup>National University of Singapore

<div align="center">
<img src="https://drive.google.com/uc?export=view&id=1lc1uIomdaS7JuHjiHdA07WO8AJETs0TM" alt="DCS Flow"  width="800"/>
</div>


## 📄 Abstract

Language models are strong few-shot learners and achieve good overall accuracy in text classification tasks, masking the fact that their results suffer from great class accuracy imbalance. We believe that the pursuit of overall accuracy should not come from enriching the strong classes, but from raising up the weak ones. To address the imbalance, we propose a **Heaviside step function** based ensemble debiasing method, which enables flexible rectifications of in-context learned class probabilities at both class and sample levels. Evaluations with Llama-2-13B on seven text classification benchmarks show that our approach achieves state-of-the-art overall accuracy gains with balanced class accuracies. More importantly, we perform analyses on the resulted probability correction scheme, showing that sample-level corrections are necessary to **elevate weak classes**. Due to effectively correcting weak classes, our method also brings significant performance gains to a larger model variant, Llama-2-70B, especially on a biomedical domain task, further demonstrating the necessity of ensemble debiasing at both levels.

---

## 📋 Get Started!

This repository contains the implementation of the **DCS** paper, providing post-hoc ICL probability correction that directly mitigates class accuracy imbalance in LLM predictions.

For experiment replications, obtain Hugging Face Llama-2 models for evaluations on Llama-2-13B and Llama-2-70B (https://huggingface.co/meta-llama).

Prerequisites: **sklearn, torch, and transformers (you probably already have these)**. If not, you may create an environment and install the exact packages from the paper.


```bash
conda env create -f environment.yml
conda activate dcs
```

---

## ✈️ Usage

### Rectify ICL Ouput Probabilities with Correction Indices

DCS dynamically chooses the correction type of each output ICL class for you. The optimization process returns correction indices that map to either a weight or a membership function. At inference, simply plug in the indices to correct.


#### 📘 Run Experiments

Run `dcs.py` to obtain experimental results on reducing class accuracy differences and enhancing overall accuracy. Our paper exprimented with three ICL settings: 1-shot, 5-shot, and N-shot. Use `vectors` directory to obtain initial ICL outputs.

```bash
python dcs.py \
		-c config/default_params.json \
		--vec_dir vectors/llama2-13b/ddi_llama2-13b_shot1_seed1
```

**🔴 Required Arguments:**
- `-c`: Default parameters for simulated annealing and objective function
- `--vec_dir`: Initial ICL output class probabilities to be debiased


---

#### 💻 CLI (Beta)

Adjust `dcs-cli.py` as you need for plug-in deployment and CLI usage. For example, DCS-CLI returns `([1, 13, 13, 14], 30)`, which is applied at inference to correct ICL outpus. This core functionality is made available via:

```bash
optimal_indices, optimal_weights = deployment_cli_main(config)
```

##### ⚙️ Correction Index Mapping

The mapping table illustrated below contains **19 triangular membership functions** for sample-level correction (**F**) and **30-point weight scale** for class-level correction (**W**).

<div align="center">
<img src="https://drive.google.com/uc?export=view&id=15Q12mov5GNIAj9ApMwFkmvdGueDYTStn" alt="DCS Mapping" width="800"/>
</div>


*🔵 Important:* Beware that the correction indices used in the paper **begin at 1**, while the code **begins at 0**.



## 📚 Citation

Please cite our paper if you make use of it.

```bibtex
@inproceedings{
	lin2025ensemble,
	title={Ensemble Debiasing Across Class and Sample Levels for Fairer Prompting Accuracy},
	author={Ruixi Lin and Ziqiao Wang and Yang You},
	booktitle={Second Conference on Language Modeling},
	year={2025},
	url={https://openreview.net/forum?id=63c7hTrUCh}
}
```

---

<div align="center">
<strong> ⭐ Found this useful? Star this repo. 🚀 </strong>
</div>




