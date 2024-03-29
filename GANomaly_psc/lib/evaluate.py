""" Evaluate ROC

Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)

def evaluate(labels, scores, RGB_score, epoch4Test, ab_thres, metric='RGB'):
    if metric == 'roc':
        return roc(labels, scores, epoch4Test)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels.cpu(), scores.cpu())
    elif metric == 'RGB':
        if (epoch4Test == 1 or epoch4Test % 20 == 0):
            evaluate_RGB(labels, RGB_score, epoch4Test, ab_thres)
        return roc(labels, scores, epoch4Test)
    else:
        raise NotImplementedError("Check the evaluation metric.")

def evaluate_RGB(labels, scores, epoch4Test, ab_thres):
    plt.clf()
    labels = labels.cpu()
    mapped = [0 for i in range(len(scores))]
    tnr = []
    tpr = []
    fpr = []
    x = []

    for thres in range(3000):
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        
        for i in range(len(scores)):
            if (scores[i] >= thres / 10000): #abnormal
                if(labels[i] == 0):
                    tp += 1
                else:
                    fp += 1
            else:
                if(labels[i] == 1):
                    tn += 1
                else:
                    fn += 1
        
        tnr.append(fp / (tn+fp))
        tpr.append(tp / (tp+fn))
        fpr.append(1 - (fp / (fp+tn)))
        x.append(thres / 10000)
        if(thres/10000 == ab_thres):
            print(f'thres: {thres/10000}')
            print(f'tn: {tn}\tfp: {fp}')
            print(f'fn: {fn}\ttp: {tp}')
            print(f'f1-score: {tp / (tp+(fp+fn)/2)}')
    plt.scatter(tnr, tpr, c='red')
    plt.savefig('./output/ganomaly/casting/test/roc/' + str(epoch4Test) + '.png')
    # print('tnr\t\ttpr\t')
    # for i in range(len(tnr)):
    #     print(f'{tnr[i]: 4f}', end='')
    #     print(f'\t{tpr[i] :.4f}', end='')
    #     if(tpr[i] != 0):
    #         print(f'\t{tnr[i]/tpr[i]}')
    plt.clf()
    plt.scatter(x, tpr, c='red')
    plt.scatter(x, fpr, c='blue')
    plt.xlabel('Threshold')
    plt.ylabel('True Positive Rate')
    plt.show()
    

# ##
# def eval_roc(labels, scores, saveto=None):
#     labels = labels.cpu()
#     for thres in range()


def roc(labels, scores, epoch4Test, saveto='./',):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.clf()
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        # plt.show()
        
        if (epoch4Test == 1 or epoch4Test % 20 == 0):
            # plt.savefig(os.path.join(saveto, "ROC.pdf"))
            plt.savefig('./output/ganomaly/casting/test/roc/abscore/' + str(epoch4Test) + '.png')
        plt.close()

    return roc_auc

    # labels = labels.cpu()
    # print(roc_curve(labels, scores))

def auprc(labels, scores):
    ap = average_precision_score(labels.cpu(), scores.cpu())
    return ap
