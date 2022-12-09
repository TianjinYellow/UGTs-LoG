



import os
import time 
import torch
import random
import shutil
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

__all__ = ['test_calibration']

def extract_prediction(val_loader, model, args):

    model.eval()
    start = time.time()

    y_pred = []
    y_true = []

    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()
    
        # compute output
        with torch.no_grad():
            output = model(input)
            pred = F.softmax(output, dim=1)

        y_pred.append(pred.cpu().numpy())
        y_true.append(target.cpu().numpy())

        if i % args.print_freq == 0:
            end = time.time()
            print('Calibration: [{0}/{1}]\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start))
            start = time.time()
    
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    #print('prediction shape = ', y_pred.shape)
    #print('ground truth shape = ', y_true.shape)

    return y_pred, y_true


def expected_calibration_error(y_true, y_pred, num_bins=5):
    #print(y_true.shape)
    #print(y_pred.shape)
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)
    #print(bins)
    #print(prob_y)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]


def test_calibration(val_loader, model, args):

    y_pred, y_true = extract_prediction(val_loader, model, args)
    ece = expected_calibration_error(y_true, y_pred)
    nll = F.nll_loss(torch.from_numpy(y_pred).log(), torch.from_numpy(y_true), reduction="mean")
    print('* ECE = {}'.format(ece))
    print('* NLL = {}'.format(nll))

    return ece, nll



