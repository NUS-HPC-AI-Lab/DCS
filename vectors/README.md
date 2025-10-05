# Obtain In-Context Learning (ICL) Output Class Probabilities

---

## ⛳ Generate Probability Vectors Or Use Our Vectors

We assess the capabilities of DCS in reducing class accuracy differences and enhancing overall accuracy with three ICL settings: 1-shot, 5-shot, and N-shot. Demonstrations in the prompt are randomly selected from the optimization set for 1-shot and 5-shot cases, while the N-shot demonstration cascades examples from each class, to minimize under-representations of any class.


### 🔨 Run ICL Probability Vector Generation
Examples are found in `example_gen.sh`. Use `predict_llm.py` to obtain output class probabilities:

```bash
python predict_llm.py \
	--model /your/path/Llama-2-13b-hf \
	--dataset ddi \
	--num_seeds 3 \
	--start_seed 0 \
	--num_shots 1 \
	--bs 2 \
	--split opt \
	--gpu_id 0 
```

**🔴 Required Arguments:**
- `--model`: Path to the LLM. Currently optimized and tested for Llama2-13B/70B (matching the models used in the paper). Please open an issue if you require support for other LLMs.
- `--dataset`: The name of the evaluation dataset to run. Supported open-source datasets include `ddi`, `trec`, `dbpedia`, `agnews`, `pubmedqa`, `sst5`, and `rte`. The corresponding data directory is expected to contain standard `training` and `test` subsets.
- `--num_seeds`: The number of runs performed to generate the initial ICL output probabilities. This accounts for the variability caused by different demonstrations, and the final DCS results are averaged across these sets of initial probabilities.
- `--start_seed`: Starting seed
- `--num_shots`: Number of shots in few-shot ICL
- `--bs`: Batch size for generation
- `--split`: Either `opt` or `test`
- `--gpu_id`: GPU used to perform prompting 


---

### 📮 Use Our Vectors (Llama2-13B/70B)
Download [here](https://drive.google.com/drive/folders/1hZPLQh2Cpg_kDgSSigMiAivQO1aXVADR?usp=sharing)

---


<div align="center">
<strong> ⭐ Found this useful? Star this repo. 🚀 </strong>
</div>
