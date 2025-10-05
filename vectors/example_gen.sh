#!/bin/bash

### Run through datasets ###

# model_name='/your/path/Llama-2-13b-hf'
# datasets=('ddi' 'trec' 'dbpedia' 'agnews' 'pubmedqa' 'sst5' 'rte')
# splits=('opt' 'test')

# for split in "${splits[@]}"
# do
# 	for dataset in "${datasets[@]}"
# 	do
# 		nohup python predict_llm.py \
# 			--model ${model_name} \
# 			--dataset ${dataset} \
# 			--num_seeds 3 \
# 			--start_seed 0 \
# 			--num_shots 1 \
# 			--bs 2 \
# 			--split ${split} \
# 			--gpu_id 0 \
# 			> vec_${split}_${dataset}.log 2>&1 &
# 	done
# done


### Run a single dataset ###

python predict_llm.py \
	--model /your/path/Llama-2-13b-hf \
	--dataset ddi \
	--num_seeds 3 \
	--start_seed 0 \
	--num_shots 1 \
	--bs 2 \
	--split opt \
	--gpu_id 0

