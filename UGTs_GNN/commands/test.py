
import os
import time
import datetime
import json

import torch
from utils.seed import set_random_seed
from utils.output_manager import OutputManager
from utils.pd_logger import PDLogger
from torch.nn import DataParallel

from models.supervised_learning import SupervisedLearning

def test(exp_name, cfg, prefix="", epoch=None, use_best=False):
    set_random_seed(cfg['seed'])
    device = torch.device('cuda:0' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
    outman = OutputManager(cfg['output_dir'], exp_name)

    outman.print('Number of available gpus: ', torch.cuda.device_count(), prefix=prefix)

    if cfg['learning_framework'] == 'SupervisedLearning':
        learner = SupervisedLearning(outman, cfg, device, cfg['data_parallel'])
    else:
        raise NotImplementedError


    if use_best:
        dump_path = outman.get_abspath(prefix=f"best.{prefix}", ext="pth")
    elif epoch is not None:
        dump_path = outman.get_abspath(prefix=f'epoch{epoch}.{prefix}', ext="pth")
    else:
        dump_path = outman.get_abspath(prefix=f"dump.{prefix}", ext="pth")

    outman.print(dump_path, prefix=prefix)
    if os.path.exists(dump_path):
        try:
            dump_dict = torch.load(dump_path)
            epoch = dump_dict['epoch']
            if isinstance(learner.model, DataParallel):
                learner.model.module.load_state_dict(dump_dict['model_state_dict'])
            else:
                learner.model.load_state_dict(dump_dict['model_state_dict'])
        except Exception as e:
            print("[train.py] catched unexpected error in loading checkpoint:", str(e))
            print("[train.py] start training from scratch")
    else:
        raise Exception

    outman.print('[', str(datetime.datetime.now()) , '] Evaluate on Test Dataset...' , prefix=prefix)

    # Test
    result = learner.evaluate(dataset_type='test')
    if use_best:
        outman.print('Test Accuracy (Best):', str(result['accuracy']), prefix=prefix)
    else:
        outman.print('Test Accuracy:', str(result['accuracy']), prefix=prefix)

    test_info_dict = {
            'accuracy': result['accuracy'],
            'epoch': epoch,
            'loss': result['loss'],
            'prefix': prefix,
            }

    if use_best:
        output_path = outman.get_abspath(prefix=f"test_best.{prefix}", ext="json")
    else:
        output_path = outman.get_abspath(prefix=f"test_epoch{epoch}.{prefix}", ext="json")

    with open(output_path, 'w') as f:
        json.dump(test_info_dict, f, indent=2)

