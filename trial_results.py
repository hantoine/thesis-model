import os
import torch
import pandas as pd
import numpy as np
from metrics import best_precision_recall_f1, \
                    area_under_precision_recall_gain_curve, \
                    area_under_roc_curve, \
                    percent_actual_positive

class TrialResults:
    def __init__(self, hparams):
        self.results = {
            'test': [],
            'val': []
        }
        self.used_ids = set()
        self.hparams = hparams
        self.agg_results = {}

    def add(self, result, fold_id, repeat_id):
        self.check_id(fold_id, repeat_id)
        val_res, test_res = result
        self.results['val'].append({'fold_id': fold_id,
                                    'repeat_id': repeat_id,
                                    **val_res})
        self.results['test'].append({'fold_id': fold_id,
                                     'repeat_id': repeat_id,
                                     **test_res})

    def check_id(self, fold_id, repeat_id):
        identifier = (fold_id, repeat_id)
        if identifier in self.used_ids:
            raise ValueError(f"Result for fold {fold_id} and repeat {repeat_id}"
                             f" already exists")
        self.used_ids.add(identifier)

    def save(self):
        save_dir = os.path.join(self.hparams.test_tube_save_path,
                                self.hparams.experiment_name)
        for set_type in ('test', 'val'):
            self.save_results(self.results[set_type], save_dir, f'{set_type}_results.csv')
        agg_val_res = self.aggregate(on_set='val')
        agg_test_res = self.aggregate(on_set='test')
        trial_results_summary = pd.DataFrame([agg_val_res, agg_test_res]).T
        trial_results_summary.columns = ['val', 'test']
        trial_results_summary.to_csv(os.path.join(save_dir, 'res_summary.csv'))

    @staticmethod
    def save_results(results, save_dir, filename):
        save_path = os.path.join(save_dir, filename)
        pd.DataFrame(results).to_csv(save_path, index=False)

    def aggregate(self, on_set='test'):
        """
            Aggregate results on different folds for each repeat and return
            the aggregated results of the best repeat.
            Aggregation for f1 is performed following this method:
            https://sebastianraschka.com/faq/docs/computing-the-f1-score.html
        """
        if not on_set in self.results:
            raise ValueError(f"Invalid set type ({on_set})")

        if on_set in self.agg_results:
            return self.agg_results[on_set]

        results = pd.DataFrame(self.results[on_set])

        repeats_results = results.groupby('repeat_id')
        tp, fp, fn, tn = repeats_results[['tp', 'fp', 'fn', 'tn']].sum().T.values

        f1 = (2 * tp) / (2 * tp + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        p_pos = (tp + fn) / (tp + fn + fp + tn)

        auc_roc = repeats_results.auc_roc.median()
        auc_prg = repeats_results.auc_prg.median()
        best_f1 = repeats_results.f1.mean()

        # Aggregation using raw predictions
        file_format = os.path.join(self.hparams.test_tube_save_path,
                                  self.hparams.experiment_name,
                                  'version_{}/eval_predictions/{}_{}_{}.npy')

        agg_results = []
        for i, repeat_result in results.groupby('repeat_id'):
            predictions = []
            targets = []
            for epoch, exp_ver in repeat_result[['epoch', 'exp_ver']].values:
                predictions.append(np.load(
                    file_format.format(exp_ver, on_set, 'prediction', epoch)))
                targets.append(np.load(
                    file_format.format(exp_ver, on_set, 'target', epoch)))
            targets = torch.from_numpy(np.concatenate(targets))
            predictions = torch.from_numpy(np.concatenate(predictions))
            agg_result = {}
            agg_result.update(best_precision_recall_f1(predictions, targets))
            agg_result.update(percent_actual_positive(predictions, targets))
            agg_result.update(
                area_under_precision_recall_gain_curve(predictions, targets))
            agg_result.update(area_under_roc_curve(predictions, targets))
            agg_result = {k: v.item() for k, v in agg_result.items()}
            agg_results.append(agg_result)
        agg_results = pd.DataFrame(agg_results)

        results = {'f1_0.25_agg': f1,
                   'prec_0.25_agg': prec,
                   'rec_0.25_agg': rec,
                   'p_pos_0.25_agg': p_pos,
                   'f1_best_mean': best_f1,
                   'auc_roc_med': auc_roc,
                   'auc_prg_med': auc_prg,
                   'f1_best_agg': agg_results.f1,
                   'prec_best_agg': agg_results.prec,
                   'rec_best_agg': agg_results.rec,
                   'p_pos_best_agg': agg_results.p_pos,
                   'thres_best_agg': agg_results.threshold,
                   'auc_roc_agg': agg_results.auc_roc,
                   'auc_prg_agg': agg_results.auc_prg
                  }
        best_repeat = agg_results.auc_prg.idxmax()

        final_results = {}
        final_results.update({k: v[best_repeat] for k, v in results.items()})
        final_results.update({f"{k}_std": v.std() for k, v in results.items()})
        final_results['n_repeats'] = len(repeats_results)
        self.agg_results[on_set] = final_results
        return final_results


