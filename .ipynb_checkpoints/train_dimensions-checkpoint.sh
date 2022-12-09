#!/bin/bash
source activate GNN
#datasets:Cora, Citeseer, Pubmed
#train_mode: score_only, normal
#for l in 2 3 4 5 6 7 8
# for dim in 16 32 64 128 256 512 768 1024 
#for s in 0 0.01 0.03 0.05 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99
#for s in 0
# for h in 1 2 3 4 5 6 7 8 9 10
#S=0.01

#dataset=Cora
dataset=Citeseer
#dataset=Pubmed
#dataset=ogbn-arxiv

#model=GCN
model=GAT
#model=GIN

#mode=score_only
mode=normal
hidden=448

for l in 32
do
    #for S in 0.01 0.03 0.05 0.07 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99
    for S in 0.45
    do
    python main.py --command train  --num_layers $l --dim_hidden $hidden --dataset $dataset --heads 1  --train_mode $mode --linear_sparsity $S --rerand_freq 0 --exp_name test10 --rerand_lambda 0.1  --epochs 400  --type_model $model  --weight_l1 0  --repeat_times 1 --sparse_decay   --random_seed 3300 --weight_decay 0.0
    done
done