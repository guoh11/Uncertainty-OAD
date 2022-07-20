import os
import json
from collections import OrderedDict

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

__all__ = [
    'compute_result_multilabel',
    'compute_result',
]

def compute_result_multilabel(class_index, score_metrics, target_metrics, save_dir, result_file,
                              ignore_class=[0], save=True, verbose=False, smooth=False, switch=False):
    result = OrderedDict()
    score_metrics = np.array(score_metrics)
    pred_metrics = np.argmax(score_metrics, axis=1)
    target_metrics = np.array(target_metrics)

    ###################################################################################################################
    # We follow (Shou et al., 2017) and adopt their per-frame evaluation method of THUMOS'14 datset.
    # Source: https://bitbucket.org/columbiadvmm/cdc/src/master/THUMOS14/eval/PreFrameLabeling/compute_framelevel_mAP.m
    ###################################################################################################################

    # Simple temporal smoothing via NMS of 5-frames window
    if smooth:
        prob = np.copy(score_metrics)
        prob1 = prob.reshape(1, prob.shape[0], prob.shape[1])
        prob2 = np.append(prob[0, :].reshape(1, -1), prob[0:-1, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob3 = np.append(prob[1:, :], prob[-1, :].reshape(1, -1), axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob4 = np.append(prob[0:2, :], prob[0:-2, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob5 = np.append(prob[2:, :], prob[-2:, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        probsmooth = np.squeeze(np.max(np.concatenate((prob1, prob2, prob3, prob4, prob5), axis=0), axis=0))
        score_metrics = np.copy(probsmooth)

    # Assign cliff diving (5) as diving (8)
    if switch:
        switch_index = np.where(score_metrics[:, 5] > score_metrics[:, 8])[0]
        score_metrics[switch_index, 8] = score_metrics[switch_index, 5]

    # Remove ambiguous (21)
    valid_index = np.where(target_metrics[:, 21]!=1)[0]

    # Compute AP
    result['AP'] = OrderedDict()
    for cls in range(len(class_index)):
        if cls not in ignore_class:
            result['AP'][class_index[cls]] = average_precision_score(
                (target_metrics[valid_index, cls]==1).astype(np.int),
                score_metrics[valid_index, cls])
            if verbose:
                print('{} AP: {:.5f}'.format(class_index[cls], result['AP'][class_index[cls]]))

    # Compute mAP
    result['mAP'] = np.mean(list(result['AP'].values()))
    if verbose:
        print('mAP: {:.5f}'.format(result['mAP']))

    # Save
    if save:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, result_file), 'w') as f:
            json.dump(result, f)
        if verbose:
            print('Saved the result to {}'.format(os.path.join(save_dir, result_file)))

    return result['mAP']

def compute_result(class_index, score_metrics, target_metrics, save_dir, result_file, ignore_class=[0], save=True, verbose=False):
    result = OrderedDict()
    score_metrics = np.array(score_metrics)
    pred_metrics = np.argmax(score_metrics, axis=1)
    target_metrics = np.array(target_metrics)

    # Compute ACC
    correct = np.sum((target_metrics!=0) & (target_metrics==pred_metrics))
    total = np.sum(target_metrics!=0)
    result['ACC'] = correct / total
    if verbose:
        print('ACC: {:.5f}'.format(result['ACC']))

    # Compute confusion matrix
    result['confusion_matrix'] = confusion_matrix(target_metrics, pred_metrics).tolist()

    # Compute AP
    result['AP'] = OrderedDict()
    for cls in range(len(class_index)):
        if cls not in ignore_class:
            result['AP'][class_index[cls]] = average_precision_score(
                (target_metrics==cls).astype(np.int),
                score_metrics[:, cls])
            if verbose:
                print('{} AP: {:.5f}'.format(class_index[cls], result['AP'][class_index[cls]]))

    # Compute mAP
    result['mAP'] = np.mean(list(result['AP'].values()))
    if verbose:
        print('mAP: {:.5f}'.format(result['mAP']))

    # Save
    if save:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, result_file), 'w') as f:
            json.dump(result, f)
        if verbose:
            print('Saved the result to {}'.format(os.path.join(save_dir, result_file)))

    return result['mAP']
