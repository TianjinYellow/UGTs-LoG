# UGTs Official Pytorch implementation

<img src="https://github.com/TienjinHuang/UGTs-LoG/blob/main/all_sparsity.png" width="800" height="300">

**You Can Have Better Graph Neural Networks by Not Training Weights at All: Finding Untrained GNNs Tickets**<br>
Tianjin Huang, Tianlong Chen, Meng Fang, Vlado Menkovski, Jiaxu Zhao, Lu Yin, Yulong Pei, Decebal Constantin Mocanu, Zhangyang Wang, Mykola Pechenizkiy, Shiwei Liu<br>
https://arxiv.org/pdf/2211.15335.pdf<br>

Abstract: *Recent works have impressively demonstrated that there exists a subnetwork in randomly initialized convolutional neural networks (CNNs) that can match the performance of the fully trained dense networks at initialization, without any optimization of the weights of the network (i.e., untrained networks). However, the presence of such untrained subnetworks in graph neural networks (GNNs) still remains mysterious. In this paper we carry out the first-of-its-kind exploration of discovering matching untrained GNNs. With sparsity as the core tool, we can find untrained sparse subnetworks at the initialization, that can match the performance of fully trained dense GNNs. Besides this already encouraging finding of comparable performance, we show that the found untrained subnetworks can substantially mitigate the GNN over-smoothing problem, hence becoming a powerful tool to enable deeper GNNs without bells and whistles. We also observe that such sparse untrained subnetworks have appealing performance in out-of-distribution detection and robustness of input perturbations. We evaluate our method across widely-used GNN architectures on various popular datasets including the Open Graph Benchmark (OGB).*

This code base is created by Tianjin Huang [t.huang@tue.nl](mailto:t.huang@tue.nl) during his Ph.D. at Eindhoven University of Technology.<br>
The implementation is heavily based on Daiki Chijiwa' implemenation for experiments on the [iterand] (https://github.com/dchiji-ntt/iterand).


## Requirements

pip install rdkit-pypi cython<br>
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia<br>
pip install ogb==1.3.1 tqdm<br>
conda install pyg -c pyg -c conda-forge<br>
pip install pyyaml==5.3.1 pandas==1.2.0<br>
conda install -c dglteam dgl-cuda11.6<br>
pip install littleballoffur<br>

## Usage

### On Cora with GAT

python main.py --command train  --lr 0.01 --num_layers 2 --dim_hidden 256 --dataset Cora --heads 1 --rerand_freq 0 --train_mode score_only --linear_sparsity 0.95 --exp_name test10  --epochs 400  --type_model GAT  --weight_l1 0  --repeat_times 1 --sparse_decay --weight_decay 0.0 --random_seed 100

### On Cora with GCN

python main.py --command train  --lr 0.01 --num_layers 2 --dim_hidden 256 --dataset Cora --heads 1  --train_mode score_only --linear_sparsity 0.9  --exp_name test10  --epochs 400  --type_model GCN  --weight_l1 0  --repeat_times 1 --sparse_decay --weight_decay 0.0 

### On Cora with GCN 32 layers

python main.py --command train  --lr 0.005 --num_layers 32 --dim_hidden 448 --dataset Cora --heads 1  --train_mode score_only --linear_sparsity 0.04  --exp_name test10  --epochs 400  --type_model GCN  --weight_l1 0  --repeat_times 1 --sparse_decay --weight_decay 0.0 --random_seed 100



## Citation
If you use this library in a research paper, please cite this repository.
```
@inproceedings{
huang2022you,
title={You Can Have Better Graph Neural Networks by Not Training Weights at All: Finding Untrained {GNN}s Tickets},
author={Tianjin Huang and Tianlong Chen and Meng Fang and Vlado Menkovski and Jiaxu Zhao and Lu Yin and Yulong Pei and Decebal Constantin Mocanu and Zhangyang Wang and Mykola Pechenizkiy and Shiwei Liu},
booktitle={The First Learning on Graphs Conference},
year={2022},
url={https://openreview.net/forum?id=dF6aEW3_62O}
}
