
import torch
import argparse
import commands
import yaml
from pprint import PrettyPrinter
import numpy as np
from base_options import BaseOptions
hyperparam_names = {
    'lr': float,
    'weight_decay': float,
    'model.config_name': str,
    'seed': int,
    'conv_sparsity': float,
    'rerand_freq': int,
    'rerand_lambda': float,
}

def load_configs(config):
    with open(config, 'r') as f:
        yml = f.read()
        dic = yaml.load(yml, Loader=yaml.FullLoader)
    return dic

def main(args):
    pp = PrettyPrinter(indent=1)
    command = args.command
    print(args)
    #print(cfg)
    command = getattr(getattr(commands, command), command)
    results_acc=[]
    results_ece=[]
    for i in range(args.repeat_times):
        acc,ece=command(args.exp_name, args=args,idx=i)
        results_acc.append(acc)
        results_ece.append(ece)
    print("acc mean:",np.mean(results_acc)," acc std:",np.std(results_acc),"ece mean:",np.mean(results_ece),"ece std",np.std(results_ece))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Constrained learing')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200,help="number of training the one shot model")
    parser.add_argument('--dim_hidden', type=int, default=64)
    parser.add_argument('--repeat_times', type=int, default=5)
    parser.add_argument('--linear_sparsity',type=float,default=0.1)
    parser.add_argument('--train_mode',type=str,default='normal',choices=['score_only','normal'])
    parser.add_argument('--exp_name', type=str,default='test', help='specify the name of experiment')
    parser.add_argument('--weight_l1',type=float, default=5e-4)
    parser.add_argument('--sampling',type=float, default=None)
    parser.add_argument('--samplingtype',type=str, default=None, choices=['RandomNodeSampler','DegreeBasedSampler','RandomEdgeSampler'])
    parser.add_argument('--sparse_decay',action='store_true')
    parser.add_argument('--attack',type=str, default=None,choices=['features', 'edges'])
    parser.add_argument('--auroc',action='store_true')
    parser.add_argument('--attack_eps',type=float, default=0)
    args = BaseOptions().initialize(parser)
    main(args)
