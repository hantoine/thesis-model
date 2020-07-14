"""
    Classes of model for estimation of accident risk from driving data
"""
import os
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
import numpy as np
from math import floor, ceil

from driving_dataset import DrivingDataset
from metrics import area_under_precision_recall_gain_curve, best_precision_recall_f1, \
                    precision_at_p_pred_pos, \
                    confusion_matrix_at_threshold, \
                    percent_actual_positive, \
                    area_under_roc_curve
from focal_loss import FocalLoss

class SeqConv(pl.LightningModule):
    """
        Applies the same conv layers to all windows, then merge windows vectors by concatenation.

        Input: (batch_size, n_windows, in_channels, window_length)
    """

    def __init__(self, hparams, k_fold_conf):
        super(SeqConv, self).__init__()

        self.hparams = hparams

        self.__check_k_fold_conf(k_fold_conf)
        self.k_fold_conf = k_fold_conf

        self.win_block = SequenceWise(WindowBlock(hparams, n_blocks=1))
        self.final_classifier = nn.Linear(hparams.n_filters, 1)

        self.criterion = FocalLoss(smooth_eps=hparams.smooth_eps)

    @staticmethod
    def __check_k_fold_conf(conf):
        train_set, val_set, test_set = set(conf['train']), set(conf['val']), set(conf['test'])
        if (len(train_set.union(val_set, test_set)) != 20
                or train_set.intersection(val_set)
                or val_set.intersection(test_set)
                or test_set.intersection(train_set)):
            raise ValueError('Invalid k-fold configuration (overlapping or some fold missing)')

    def forward(self, inputs): # pylint: disable=W0221
        out = self.win_block(inputs) # Outputs (batch_size, n_windows, n_filters)
        out = torch.mean(out, dim=1) # average over sequences
        out = self.final_classifier(out)
        out = torch.sigmoid(out)

        return out

    def training_step(self, batch, _): # pylint: disable=W0221
        input_data, target = batch
        prediction = self.forward(input_data)

        return {'loss': self.criterion.forward(prediction, target)}

    def validation_step(self, batch, _): # pylint: disable=W0221
        return self.eval_step(batch)
    def validation_end(self, outputs):
        return self.eval_end(outputs)
    def test_step(self, batch, _): # pylint: disable=W0221
        return self.eval_step(batch)
    def test_end(self, outputs):
        return self.eval_end(outputs, test=True)

    def eval_step(self, batch):
        """Evaluation step used for both validation and test """
        input_data, target, index = batch
        prediction = self.forward(input_data)

        return {'val_loss': self.criterion.forward(prediction, target),
                'prediction': prediction,
                'target': target.char(),
                'index': index}

    def eval_end(self, outputs, test=False):
        """Evaluation end function used for both validation and test """
        metric_values = {}
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        metric_values.update({'val_loss': avg_loss})

        prediction = torch.cat([x['prediction'] for x in outputs])
        target = torch.cat([x['target'] for x in outputs]).int()
        index = torch.cat([x['index'] for x in outputs]).int()

        metrics = (best_precision_recall_f1,
                   area_under_precision_recall_gain_curve,
                   precision_at_p_pred_pos(0.40),
                   confusion_matrix_at_threshold(0.25),
                   percent_actual_positive,
                   area_under_roc_curve)
        for metric in metrics:
            metric_values.update(metric(prediction, target))

        # Saving prediction for future error analysis
        save_dir = os.path.join(self.logger.experiment.save_dir, self.logger.experiment.name,
                                f"version_{self.logger.experiment.version}", 'eval_predictions')
        prefix = 'test' if test else 'val'
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/{prefix}_prediction_{self.current_epoch}.npy", prediction.cpu().numpy())
        np.save(f"{save_dir}/{prefix}_target_{self.current_epoch}.npy", target.cpu().numpy())
        np.save(f"{save_dir}/{prefix}_index_{self.current_epoch}.npy", index.cpu().numpy())
        # Used to retrieve prediction filename from results
        metric_values['epoch'] = np.array(self.current_epoch)
        metric_values['exp_ver'] = np.array(self.logger.experiment.version)

        return {
            'log': metric_values,
            'progress_bar': {k: metric_values[k] for k in ('val_loss', 'auc_prg', 'f1', 'rec')},
            **metric_values
        }

    def configure_optimizers(self):
        return [torch.optim.AdamW(self.parameters(),
                                  lr=self.hparams.learning_rate,
                                  weight_decay=self.hparams.weight_decay)]

    @pl.data_loader
    def train_dataloader(self):
        dataset = DrivingDataset(self.hparams.dataset_path, folds=self.k_fold_conf['train'],
                                 selected_parameters=self.hparams.selected_channels,
                                 in_memory=False, acc_types=self.hparams.acc_types,
                                 include_index=False)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers)

    @pl.data_loader
    def val_dataloader(self):
        dataset = DrivingDataset(self.hparams.dataset_path, folds=self.k_fold_conf['val'],
                                 selected_parameters=self.hparams.selected_channels,
                                 in_memory=False, acc_types=self.hparams.acc_types,
                                 include_index=True, normalize_like=self.train_dataloader().dataset)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers)

    @pl.data_loader
    def test_dataloader(self):
        dataset = DrivingDataset(self.hparams.dataset_path, folds=self.k_fold_conf['test'],
                                 selected_parameters=self.hparams.selected_channels,
                                 in_memory=False, acc_types=self.hparams.acc_types,
                                 include_index=True, normalize_like=self.train_dataloader().dataset)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :return:
        """
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

        # network params
        parser.opt_list('--in_channels', default=6, type=int)
        parser.opt_list('--seq_len', default=20, type=int)
        parser.opt_list('--kernel_size_conv1', default=31, type=int)
        parser.opt_list('--kernel_size_conv2', default=8, type=int)
        parser.opt_list('--kernel_size_conv3', default=4, type=int)
        parser.opt_list('--stride_conv1', default=2, type=int)
        parser.opt_list('--stride_conv2', default=2, type=int)
        parser.opt_list('--stride_conv3', default=1, type=int)
        parser.opt_list('--n_filters', default=10, options=[8, 12, 16, 20, 24, 28, 32],
                        type=int, tunable=True)
        parser.opt_list('--drop_prob', default=0.57, options=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        type=float, tunable=True)

        # data params
        parser.add_argument('--num_workers', default=5, type=int)
        default_dataset_name = '18m_road_seq-60w_any-acc-type_whole-year_no-overlap'
        parser.add_argument('--dataset_path',
                            default=os.path.join('/media/raid', default_dataset_name),
                            type=str)
        parser.add_argument('--selected_channels', type=str, default='0 1 2 3 4 5')
        parser.add_argument('--acc_types', type=int, nargs='+',
                            default=(6, 16, 8, 1, 22, 15, 14, 21, 3, 0, 7))
        # This default value means accidents of the following types are used as labels
        #   - Équipement endommagé pendant le chargement
        #   - En reculant
        #   - Frappé un objet stationnaire (excepté mur)
        #   - Pont / Viaduc
        #   - Resté pris/remorquage
        #   - Erreur d'accouplement
        #   - Frappé un vehicle stationné
        #   - Fil / Cable
        #   - Divers
        #   - Feu
        #   - Frappé mur / batisse

        # training params
        parser.opt_list('--batch_size', default=32, type=int,
                        options=[32, 64, 128, 256], tunable=False,
                        help='batch size will be divided over all gpus being used across all nodes')
        parser.opt_list('--learning_rate', default=5e-2, type=float,
                        options=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
                        tunable=True)
        parser.opt_list('--weight_decay', default=0.001, type=float,
                        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                        tunable=True)
        parser.add_argument('--smooth_eps', default=0.0, type=float)

        return parser

    def on_post_performance_check(self):
        print('\n') # Hack to print each epoch progress bar on a different line


class WindowBlock(nn.Module):
    """ Building block applied on each window """
    def __init__(self, hparams, n_blocks=3):
        super(WindowBlock, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(Block(hparams))

    def forward(x):
        for block in self.blocks:
            x = block(x)
        return torch.mean(x, dim=2) # average over time


class Block(nn.Module):
    """ Building block of model applied on all windows """
    def __init__(self, hparams, skip=False):
        super(Block, self).__init__()

        n_filters = hparams.n_filters
        in_channels = hparams.in_channels

        self.conv1 = nn.Conv1d(in_channels, n_filters, hparams.kernel_size_conv1,
                               hparams.stride_conv1)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.do1 = nn.Dropout2d(hparams.drop_prob) # Dropout{2,3}d invariant to feature dim

        self.conv2 = nn.Conv1d(n_filters, n_filters, hparams.kernel_size_conv2,
                               hparams.stride_conv2)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.do2 = nn.Dropout2d(hparams.drop_prob) # Dropout{2,3}d invariant to feature dim

        self.conv3 = nn.Conv1d(n_filters, n_filters, hparams.kernel_size_conv3,
                               hparams.stride_conv3)
        self.bn3 = nn.BatchNorm1d(n_filters)
        self.do3 = nn.Dropout2d(hparams.drop_prob)

    def forward(self, inputs): # pylint: disable=W0221

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = F.elu(out)
        out = self.do1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.elu(out)
        out = self.do2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.elu(out)
        out = self.do3(out)

        return out

class SequenceWise(nn.Module):
    """
        Collapses input of dim (T,N, *dims) to ((T*N), *dims), and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
    """
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x): # pylint: disable=W0221
        batch_size, seq_length = x.size(0), x.size(1)

        x = x.view(batch_size * seq_length, *x.size()[2:])
        x = self.module(x)
        x = x.view(batch_size, seq_length, *x.size()[1:])

        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr
