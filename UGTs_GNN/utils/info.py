#!/usr/bin/python3
import os
import json
import argparse

filter_keys = [
        'model.config_name',
        'conv_sparsity',
        'rerand_lambda',
        'lr',
        'seed',
        ]

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('-l', '--loss_train', action='store_true')
for key in filter_keys:
    parser.add_argument('--' + key, type=str, default=None)
args = parser.parse_args()
args_dic = vars(args)

def filter(info):
    ret = False
    for key in filter_keys:
        if args_dic[key] is None:
            pass
        elif key + "_" + args_dic[key] + "--" in info['prefix']:
            pass
        else:
            ret = True
    return ret

dir = args.dir
infos = []
for fn in os.listdir(dir):
    full_fn = os.path.join(dir, fn)
    if not os.path.isfile(full_fn):
        continue
    if fn[:4] == 'info':
        with open(full_fn, 'r') as f:
            infostr = f.read()
            infos.append(json.loads(infostr))


if args.loss_train:
    infos = [x for x in infos if 'acc_train' in x]
    last_infos = sorted(infos, key=lambda x: -x['acc_train'])
    print("")
    print("Sorted by last train accs:")
    for info in last_infos:
        if filter(info):
            continue
        print("%.4f (%d)\t%s" % (info['acc_train'], info['epoch'], info['prefix']))

    last_infos = sorted(infos, key=lambda x: -x['loss_train'])
    print("")
    print("Sorted by train loss:")
    for info in last_infos:
        if filter(info):
            continue
        print("%.10f (%d)\t%s" % (info['loss_train'], info['epoch'], info['prefix']))
else:
    accs = []
    best_infos = sorted(infos, key=lambda x: -x['best_val'])
    print("")
    print("Sorted by best val accs:")
    for info in best_infos:
        if filter(info):
            continue
        print("%.4f (%d/%d)\t%s" % (info['best_val'], info['best_epoch'], info['epoch'], info['prefix']))
        accs.append(info['best_val'])
    mean = sum(accs) / len(accs)
    print("Mean:", mean)


    accs = []
    last_infos = sorted(infos, key=lambda x: -x['last_val'])
    print("")
    print("Sorted by last val accs:")
    for info in last_infos:
        if filter(info):
            continue
        print("%.4f (%d)\t%s" % (info['last_val'], info['epoch'], info['prefix']))
        accs.append(info['last_val'])
    mean = sum(accs) / len(accs)
    print("Mean:", mean)
