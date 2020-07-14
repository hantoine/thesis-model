
""" Module containing metric implementations """
from prg import prg
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score

def area_under_precision_recall_gain_curve(prediction, target):
    """
        Compute the area under the precision recall gain curve.
        This curve corresponds to the precision gain drawn as a function of the recall gain.
        Precision gain and recall gain are precision and recall rescaled between their theoretical
        minimum and maximum on an harmonic scale.
        Cf: https://papers.nips.cc/paper/5867-precision-recall-gain-curves-pr-analysis-done-right
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prg_curve = prg.create_prg_curve(target.cpu().numpy().ravel(),
                                         prediction.cpu().numpy().ravel())
        auprg = prg.calc_auprg(prg_curve)
    return {'auc_prg': auprg}

def precision_at_p_pred_pos(p_pred_pos):
    def precision(prediction, target):
        threshold = prediction.sort(axis=0).values[int((1-p_pred_pos)*len(prediction))][0].item()
        predicted_class = (prediction > threshold).int()
        true_positive = (target & predicted_class).sum()
        predicted_positive = predicted_class.sum()
        positive = target.sum().float() / target.size(0)

        precision = true_positive.float() / predicted_positive
        precision_gain = (precision - positive) / ((1 - positive) * precision)
        risk_factor = precision / positive

        str_p = f'{100*p_pred_pos:.0f}p'
        return {f'prec_{str_p}': precision,
                f'precg_{str_p}': precision_gain,
                f'rf_{str_p}': risk_factor}

    return precision

def area_under_roc_curve(pred, target):
    return {'auc_roc': roc_auc_score(target.cpu(), pred.cpu())}

def confusion_matrix_at_threshold(threshold):
    """
        Useful to compute F1 accross k-folds
        (cf https://www.hpl.hp.com/techreports/2009/HPL-2009-359.pdf)
    """
    def confusion_matrix(prediction, target):
        predicted_class = (prediction > threshold).bool()
        target = target.bool()

        false_positive = (~target & predicted_class).sum()
        false_negative = (target & ~predicted_class).sum()
        true_positive = (target & predicted_class).sum()
        true_negative = (~target & ~predicted_class).sum()

        return {'tp': true_positive,
                'fp': false_positive,
                'fn': false_negative,
                'tn': true_negative}
    return confusion_matrix

def percent_actual_positive(_, target):
    "Compute the percentage of positive in this evaluation set "
    actual_positive = target.sum()
    p_pos = actual_positive.float() / len(target)
    return {'p_pos': p_pos}

def best_precision_recall_f1(prediction, target):
    """
        Compute maximal f1 score with respect to the decision threshold and associated precision
        and recall.
    """
    actual_positive = target.sum()

    def score_with_threshold(threshold):
        predicted_class = (prediction > threshold).int()

        true_positive = (target & predicted_class).sum()
        predicted_positive = predicted_class.sum()

        precision = true_positive.float() / predicted_positive
        recall = true_positive.float() / actual_positive
        f1_score = precision * recall * 2 / (precision + recall)

        return f1_score, {'prec': precision,
                          'rec': recall,
                          'f1': f1_score,
                          'threshold': np.array(threshold)}

    center = 0.5
    width = 1.0
    factor = 2.1
    found = False
    for _ in range(100):
        low = score_with_threshold(center - (width / factor))[0]
        mid, scores = score_with_threshold(center)
        high = score_with_threshold(center + (width / factor))[0]
        if low == mid and mid == high:
            found = True
            break
        max_f1 = max(low, mid, high)
        if max_f1 == low:
            center -= (width / factor)
        elif max_f1 == high:
            center += (width / factor)
        width /= 2

    if not found:
        print('Warning: best f1 not found')

    return scores
