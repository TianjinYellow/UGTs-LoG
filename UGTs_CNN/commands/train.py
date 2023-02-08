
import os
import time
import datetime
import json

import torch
from utils.seed import set_random_seed
from utils.output_manager import OutputManager
from utils.pd_logger import PDLogger
from torch.nn import DataParallel
from commands.test import test
from torch.optim.lr_scheduler import MultiStepLR
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

def get_sparsity(sparsity,current_epoches,start_epoches,end_epoches):
    if current_epoches<=end_epoches:
        sparsity1=sparsity-sparsity*(1-(current_epoches-start_epoches)*1.0/(end_epoches-start_epoches))
    else:
        sparsity1=sparsity
    return sparsity1

def train(exp_name, cfg, prefix=""):
    if cfg['seed'] is not None:
        set_random_seed(cfg['seed'])
    elif cfg['seed_by_time']:
        set_random_seed(int(time.time() * 1000) % 1000000)
    else:
        raise Exception("Set seed value.")
    device = torch.device('cuda:0' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
    outman = OutputManager(cfg['output_dir'], exp_name)

    dump_path = outman.get_abspath(prefix=f"dump.{prefix}", ext="pth")

    outman.print('Number of available gpus: ', torch.cuda.device_count(), prefix=prefix)

    pd_logger = PDLogger()
    pd_logger.set_filename(outman.get_abspath(prefix=f"pd_log.{prefix}", ext="pickle"))
    if os.path.exists(pd_logger.filename) and not cfg['force_restart']:
        pd_logger.load()

    if cfg['learning_framework'] == 'SupervisedLearning':
        learner = SupervisedLearning(outman, cfg, device, cfg['data_parallel'])
    else:
        raise NotImplementedError

    params_info = count_params(learner.model)

    best_value = None
    best_epoch = 0
    start_epoch = 0
    total_iters = 0
    total_seconds = 0.

    outman.print(dump_path, prefix=prefix)
    if os.path.exists(dump_path) and not cfg['force_restart']:
        try:
            dump_dict = torch.load(dump_path)
            start_epoch = dump_dict['epoch'] + 1
            total_iters = dump_dict['total_iters']
            best_value = dump_dict['best_val']
            best_epoch = dump_dict['best_epoch'] if 'best_epoch' in dump_dict else 0
            total_seconds = dump_dict['total_seconds'] if 'total_seconds' in dump_dict else 0.
            if isinstance(learner.model, DataParallel):
                learner.model.module.load_state_dict(dump_dict['model_state_dict'])
            else:
                learner.model.load_state_dict(dump_dict['model_state_dict'])
            learner.optimizer.load_state_dict(dump_dict['optim_state_dict'])
            if 'sched_state_dict' in dump_dict:
                learner.scheduler.load_state_dict(dump_dict['sched_state_dict'])
        except Exception as e:
            print("[train.py] catched unexpected error in loading checkpoint:", str(e))
            print("[train.py] start training from scratch")
    elif cfg['load_checkpoint_path'] is not None:
        assert not os.path.exists(dump_path)
        assert os.path.exists(cfg['load_checkpoint_path'])
        try:
            checkpoint_dict = torch.load(cfg['load_checkpoint_path'])
            if isinstance(learner.model, DataParallel):
                learner.model.module.load_state_dict(checkpoint_dict['model_state_dict'])
            else:
                learner.model.load_state_dict(checkpoint_dict['model_state_dict'])
            #learner.optimizer.load_state_dict(checkpoint_dict['optim_state_dict'])
            #if 'sched_state_dict' in checkpoint_dict:
            #    learner.scheduler.load_state_dict(checkpoint_dict['sched_state_dict'])
        except Exception as e:
            print("[train.py] catched unexpected error in loading checkpoint:", str(e))
            print("[train.py] start training from scratch")

    # Define re-randomize callback
    if cfg['rerand_mode'] is not None and cfg['rerand_freq'] > 0:
        if cfg['rerand_freq_unit'] == 'epoch':
            def rerand_callback(model, epoch, it, iters_per_epoch):
                real_model = model.module if isinstance(model, DataParallel) else model
                if (it + 1) % int(iters_per_epoch / cfg['rerand_freq']) == 0:
                    outman.print(f'[Train] rerandomized@{it}', prefix=prefix)
                    real_model.rerandomize(cfg['rerand_mode'], cfg['rerand_lambda'], cfg['rerand_mu'])
                else:
                    pass
        elif cfg['rerand_freq_unit'] == 'iteration':
            def rerand_callback(model, epoch, it, iters_per_epoch):
                real_model = model.module if isinstance(model, DataParallel) else model
                if (it + 1) % int(cfg['rerand_freq']) == 0:
                    real_model.rerandomize(cfg['rerand_mode'], cfg['rerand_lambda'], cfg['rerand_mu'])
                else:
                    pass
                if (it + 1) % iters_per_epoch == 0:
                    outman.print(f'[Train] rerandomized per', cfg['rerand_freq'], 'iterations')
        else:
            raise NotImplementedError
    else:
        def rerand_callback(model, epoch, it, iters_per_epoch):
            pass

    # Training loop
    for epoch in range(start_epoch, cfg['epoch']):
        start_sec = time.time()
        if cfg['gumbelsoftmax']:
            if cfg['gumbelsoftmaxWarmup']>0 and epoch<cfg['gumbelsoftmaxWarmup']:           
                learner.model.set_gumbelsoftmax(False)
            else:
                learner.model.set_gumbelsoftmax(True)

            s,score_norm=learner.model.get_current_sparsity()
            print("current sparsity at epoch",epoch," sparsity:",s,'score norm:',score_norm)
        if cfg['train_mode']=='score_only':
            if epoch==cfg['finetuneEpoch'] and cfg['couple'] is False:
                all_len=cfg['epoch']-cfg['finetuneEpoch']
                print("total fintune epoch:",all_len)
                learner.optimizer = learner._get_optimizer('normal',lr=cfg['finetuneLr'])
                learner.scheduler =MultiStepLR(learner.optimizer, milestones=[int(all_len*0.5)+int(cfg['finetuneEpoch']),int(all_len*0.75)+int(cfg['finetuneEpoch'])], gamma=0.1)

        outman.print('[', str(datetime.datetime.now()) , '] Epoch: ', str(epoch), prefix=prefix)

        ##gradually sparse strategy
        if cfg['gradualsparse']:
            
            current_sparsity=get_sparsity(cfg['conv_sparsity'],epoch,cfg['start_epoch'],cfg['end_epoch'])
            learner.model.set_sparsity(current_sparsity)
            learner.model.sparsity=current_sparsity
            print(epoch, "  current sparsity:",current_sparsity)

        # Train
        results_train = learner.train(epoch, total_iters, before_callback=rerand_callback)
        train_accuracy = results_train['moving_accuracy']
        results_per_iter = results_train['per_iteration']
        new_total_iters = results_train['iterations']
        total_loss_train = results_train['loss']

        pd_logger.add('train_accs', [train_accuracy], index=[epoch], columns=['train-acc'])
        outman.print('Train Accuracy:', str(train_accuracy), prefix=prefix)
        if cfg['print_train_loss']:
            outman.print('Train Loss:', str(total_loss_train), prefix=prefix)

        # Evaluate
        results_eval = learner.evaluate()
        val_accuracy = results_eval['accuracy']
        pd_logger.add('val_accs', [val_accuracy], index=[epoch], columns=['val-acc'])
        outman.print('Val Accuracy:', str(val_accuracy), prefix=prefix)

        # Save train losses per iteration
        losses = [res['mean_loss'] for res in results_per_iter]
        index = list(range(total_iters, new_total_iters))
        pd_logger.add('train_losses', losses, index=index)
        # Update total_iters
        total_iters = new_total_iters

        # Flag if save best model
        if (best_value is None) or (best_value < val_accuracy):
            best_value = val_accuracy
            best_epoch = epoch
            save_best_model = True
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
        if save_best_model and cfg['save_best_model']:
            outman.save_dict(dump_dict, prefix=f"best.{prefix}", ext="pth")
        if epoch in cfg['checkpoint_epochs']:
            outman.save_dict(dump_dict, prefix=f'epoch{epoch}.{prefix}', ext='pth')

        pd_logger.save()

    if start_epoch + 1 <= cfg['epoch']:
        test(exp_name, cfg, prefix=prefix)

