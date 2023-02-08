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
parser.add_argument('--epoch', type=str, required=True)
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
test_prefix = f'test_epoch{args.epoch}'
for fn in os.listdir(dir):
    full_fn = os.path.join(dir, fn)
    if not os.path.isfile(full_fn):
        continue
    if fn[:len(test_prefix)] == test_prefix:
        with open(full_fn, 'r') as f:
            infostr = f.read()
            infos.append(json.loads(infostr))

accs = []
last_infos = sorted(infos, key=lambda x: -x['accuracy'])
print("")
print("Sorted by test accs:")
for info in last_infos:
    if filter(info):
        continue
    print("%.4f (%d)\t%s" % (info['accuracy'], info['epoch'], info['prefix']))
    accs.append(info['accuracy'])
mean = sum(accs) / len(accs)
print("Mean:", mean)

