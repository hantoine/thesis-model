"""
Runs a model on a single node across N-gpus using dataParallel
"""
import os

import numpy as np
import pandas as pd
import torch
from test_tube import HyperOptArgumentParser

from model_training import evaluate_hyperparams
from model import SeqConv

SEED = 4668
torch.manual_seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':

    # dirs
    root_dir = os.path.dirname(os.path.realpath(os.curdir))
    demo_log_dir = os.path.join(root_dir, 'model_training_logs')
    checkpoint_dir = os.path.join(demo_log_dir, 'model_weights')
    test_tube_dir = os.path.join(demo_log_dir, 'test_tube_data')

    # although we user hyperOptParser, we are using it only as argparse right now
    parent_parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)

    # gpu args
    parent_parser.add_argument('--gpus', type=str, default='-1',
                               help='how many gpus to use in the node.'
                                    ' value -1 uses all the gpus on the node')
    parent_parser.add_argument('--dist_backend', type=str, default='dp',
                               help='When using multiple GPUs set to dp or ddp')
    parent_parser.add_argument('--test_tube_save_path', type=str, default=test_tube_dir,
                               help='where to save logs')
    parent_parser.add_argument('--model_save_path', type=str, default=checkpoint_dir,
                               help='where to save model')
    parent_parser.add_argument('--experiment_name', type=str, default='pt_lightning_exp_a',
                               help='test tube exp name')
    parent_parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                               help='accumulated gradient (default: 1 no accumulated grads)')
    parent_parser.add_argument('--overfit_pct', type=float, default=0,
                               help='overfit on x%% of data for debugging')
    parent_parser.add_argument('--fast_dev_run', dest='fast_dev_run', action='store_true',
                               help='enable fast development run')
    parent_parser.add_argument('--early_stopping_patience', type=int, default=3,
                               help='Number of epochs without improvement before stopping training')
    parent_parser.add_argument('--max_nb_epochs', type=int, default=10000,
                               help='Maximal number of epochs')

    # allow model to overwrite or extend args
    parser = SeqConv.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    # run on HPC cluster
    metrics_values = evaluate_hyperparams(hyperparams)
    print(f'Validation set: \n{pd.Series(metrics_values)}')
