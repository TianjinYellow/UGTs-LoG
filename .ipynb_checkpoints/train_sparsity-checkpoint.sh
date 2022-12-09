#!/bin/bash
source activate GNN
mode=score_only
#hidden=384
dataset=Cora
for hidden in 256
do
	for depth in 2
	do
	    for model in GAT
	    do
	        for S in 0.3
	        do
	        python main.py --command train  --num_layers $depth --dim_hidden $hidden --dataset $dataset --heads 1  --train_mode $mode --linear_sparsity $S --rerand_freq 0 --exp_name test10 --rerand_lambda 0.1  --epochs 400  --type_model $model  --weight_l1 0  --repeat_times 1 --sparse_decay --lr 0.01 --weight_decay 0
	        done
	    done
	done
done

source deactivate
