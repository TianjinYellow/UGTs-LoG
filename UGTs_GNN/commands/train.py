
import os
import time
import datetime
import json
import copy
import torch
from utils.seed import set_random_seed
from utils.output_manager import OutputManager
from utils.pd_logger import PDLogger
from torch.nn import DataParallel
from commands.test import test

from models.supervised_learning import SupervisedLearning

def count_params(model):
    count = 0
    count_not_score = 0
    count_reduced = 0
    for (n,p) in model.named_parameters():
        count += p.flatten().size(0)
        if hasattr(p, 'is_score') and p.is_score:
            print(n+':', int(p.flatten().size(0) * (1.0 - p.sparsity)), '/', p.flatten().size(0), '(sparsity =', p.sparsity,')')
            count_reduced += int(p.flatten().size(0) * p.sparsity)
        else:
            print(n+':',p.flatten().size(0))
            count_not_score += p.flatten().size(0)
    count_after_pruning = count_not_score - count_reduced
    total_sparsity = 1 - (count_after_pruning / count_not_score)
    print('Params after/before pruned:\t', count_after_pruning, '/', count_not_score, '(sparsity: ' + str(total_sparsity) +')')
    print('Total Params:\t', count)
    return {
            'params_after_pruned': count_after_pruning,
            'params_before_pruned': count_not_score,
            'total_params': count,
            'sparsity': total_sparsity,
            }

def train(exp_name, args,prefix="",idx=1):
    if args.dataset=='Cora':
        seeds=[5300,2100,100,900,3500]
    elif args.dataset=='Citeseer':
        seeds=[3300,5200,5700,4900,5000]
    elif args.dataset=='Pubmed':
        seeds=[1000,2600,3100,5200,3000]
    elif args.dataset=='ogbn-arxiv':
        seeds=[2700,4000,3200]
    else:
        pass
    
    if args.random_seed is not None:
        set_random_seed(args.random_seed)
    elif args.seed_by_time:
        set_random_seed(seeds[idx])
    else:
        raise Exception("Set seed value.")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    outman = OutputManager(args.output_dir, exp_name)

    dump_path = outman.get_abspath(prefix=f"dump.{prefix}", ext="pth")

    outman.print('Number of available gpus: ', torch.cuda.device_count(), prefix=prefix)

    if args.learning_framework == 'SupervisedLearning':
        learner = SupervisedLearning(outman, args, device)
    else:
        raise NotImplementedError

    params_info = count_params(learner.model)

    best_value = None
    best_epoch = 0
    start_epoch = 0
    total_iters = 0
    total_seconds = 0.

    outman.print(dump_path, prefix=prefix)
    # Training loop
    #best_acc=0.0
    for epoch in range(start_epoch, args.epochs):
        start_sec = time.time()

        #outman.print('[', str(datetime.datetime.now()) , '] Epoch: ', str(epoch), prefix=prefix)

        # Train
        results_train = learner.train(epoch, args.epochs)
        train_accuracy = results_train['moving_accuracy']
        results_per_iter = results_train['per_iteration']
        new_total_iters = results_train['iterations']
        total_loss_train = results_train['loss']

        #pd_logger.add('train_accs', [train_accuracy], index=[epoch], columns=['train-acc'])
        #outman.print('Train Accuracy:', str(train_accuracy), prefix=prefix)
        #if args.print_train_loss:
        #    outman.print('Train Loss:', str(total_loss_train), prefix=prefix)

        # Evaluate
        val_accuracy,test_accuracy = learner.evaluate()
        #val_accuracy = results_eval['accuracy']
        #pd_logger.add('test_accs', [test_accuracy], index=[epoch], columns=['test-acc'])
        #if (epoch+1)%20==0:
        #    outman.print("epoch",str(epoch),'Train Accuracy:', str(train_accuracy),'val accuracy',str(val_accuracy),'test Accuracy:', str(test_accuracy),"L1_loss",results_train['L1_loss'], prefix=prefix)

        # Save train losses per iteration
        losses = [res['mean_loss'] for res in results_per_iter]
        index = list(range(total_iters, new_total_iters))
        #pd_logger.add('train_losses', losses, index=index)
        # Update total_iters
        total_iters = new_total_iters

        # Flag if save best model
        if (best_value is None) or (best_value < val_accuracy):
            best_value = val_accuracy
            best_epoch = epoch
            save_best_model = True
            best_states=copy.deepcopy(learner.model.state_dict())
        else:
            save_best_model = False

        end_sec = time.time()
        total_seconds += end_sec - start_sec
        if isinstance(learner.model, DataParallel):
            model_state_dict = learner.model.module.state_dict()
        else:
            model_state_dict = learner.model.state_dict()
        dump_dict = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optim_state_dict': learner.optimizer.state_dict(),
                'sched_state_dict': learner.scheduler.state_dict(),
                'best_val': best_value,
                'best_epoch': best_epoch,
                'total_iters': total_iters,
                'total_seconds': total_seconds,
        }
        info_dict = {
                'last_val': val_accuracy,
                'epoch': epoch,
                'best_val': best_value,
                'best_epoch': best_epoch,
                'loss_train': total_loss_train,
                'acc_train': train_accuracy,
                'total_time': str(datetime.timedelta(seconds=int(total_seconds))),
                'total_seconds': total_seconds,
                'prefix': prefix,
                'params_info': params_info,
        }
        outman.save_dict(dump_dict, prefix=f"dump.{prefix}", ext="pth")
        with open(outman.get_abspath(prefix=f"info.{prefix}", ext="json"), 'w') as f:
            json.dump(info_dict, f, indent=2)
        if save_best_model and args.save_best_model:
            outman.save_dict(dump_dict, prefix=f"best.{prefix}", ext="pth")
        if epoch in args.checkpoint_epochs:
            outman.save_dict(dump_dict, prefix=f'epoch{epoch}.{prefix}', ext='pth')

        #pd_logger.save()

    learner.model.load_state_dict(best_states)
    val_acc,test_acc=learner.evaluate()
    #learner.plot_tsne("./Figures/")
    #ece=learner.get_ece()
    ece=0.0
    print('best acc:', test_acc,"ece",ece)
    return test_acc,ece
    

    #if start_epoch + 1 <= cfg['epoch']:
    #    test(exp_name, cfg, prefix=prefix)

