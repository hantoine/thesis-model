import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from sklearn.model_selection import RepeatedKFold 

from model import SeqConv
from torch_lr_finder import LRFinder
from trial_results import TrialResults

def evaluate_hyperparams(hparams, verbose=True, sanity_check=False,
                        kfold_n_split=3, kfold_n_repeat=3, n_repeats=5,
                        n_test_datagroup=6, use_lr_finder=True):
    """
        Evaluate performances obtained with the given hyperparameters using
        cross-validation.
        A repeated k-fold cross-validation is performed using predefined data
        groups. Data groups have been made to contain all the data of a subset
        of drivers and to all have approximately the same portion of positives
        in order to ensure the data of a driver cannot be in both train and
        validation sets and that different folds have approximately the same
        positive rate. There are 20 predefined data groups, the last 
        n_test_datagroup are used for tests. For each k-fold trial, the neural
        network is trained n_repeats times with different random 
        initializations of its parameters. The performances on the test set
        are also computed and saved to file but not returned. Performance
        metrics on folds are aggregated to return a scalar performance score
        that can be used for tuning.
    """
    results = TrialResults(hparams)
    rkf = RepeatedKFold(n_splits=kfold_n_split, n_repeats=kfold_n_repeat)
    train_dg = np.zeros(20 - n_test_datagroup)
    for split_idx, (train_idx, val_idx) in enumerate(rkf.split(train_dg)):
        k_fold_conf = {
            'train': train_idx,
            'val': val_idx,
            'test': range(20 - n_test_datagroup, 20)
        }
        if use_lr_finder:
            use_lr_finder = False
            best_lr = find_lr(hparams, k_fold_conf)
            setattr(hparams, 'learning_rate', best_lr)
        for repeat_id in range(n_repeats):
            result = train_model(hparams, k_fold_conf, verbose,
                                 sanity_check)
            results.add(result, split_idx, repeat_id)
    results.save()
    return results.aggregate(on_set='val')


def find_lr(hparams, k_fold_conf):
    setattr(hparams, 'learning_rate', 1e-5)  # Set start lr very low
    model = SeqConv(hparams, k_fold_conf)
    lr_finder = LRFinder(model, model.configure_optimizers()[0], model.criterion, device="cuda")
    lr_finder.range_test(model.train_dataloader(), end_lr=100, num_iter=100)

    history = pd.DataFrame(lr_finder.history)
    history['loss_diff'] = history['loss'].diff().ewm(alpha=0.05).mean()
    history = history.iloc[:-5]
    plt.subplot(2, 1, 1)
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.plot(history['lr'], history['loss'])
    plt.subplot(2, 1, 2)
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Loss difference")
    plt.plot(history['lr'], history['loss_diff'])
    plot_save_dir = os.path.join(hparams.test_tube_save_path,
                                 hparams.experiment_name,
                                 'lrfinder_plot.png')
    os.makedirs(os.path.dirname(plot_save_dir), exist_ok=True)
    plt.savefig(plot_save_dir)

    history.set_index('lr', inplace=True)
    best_lr = history['loss_diff'].idxmin()
    print(f'Found best learning rate: {best_lr}')
    return best_lr


def train_model(hparams, k_fold_conf, verbose=True, sanity_check=True):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SeqConv(hparams, k_fold_conf)

    # ------------------------
    # 2 INIT TEST TUBE LOGGER
    # ------------------------

    logger = TestTubeLogger(
        name=hparams.experiment_name,
        save_dir=hparams.test_tube_save_path,
    )
    exp = logger.experiment

    # ------------------------
    # 3 DEFINE CALLBACKS
    # ------------------------
    model_save_path = '{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)
    early_stop = EarlyStopping(
        monitor='auc_prg',
        patience=hparams.early_stopping_patience,
        verbose=False,
        mode='max'
    )

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        verbose=False,
        monitor='auc_prg',
        mode='max'
    )

    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        show_progress_bar=verbose,
        max_nb_epochs=hparams.max_nb_epochs,
        logger=logger,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        gpus=hparams.gpus,
        distributed_backend=hparams.dist_backend,
        fast_dev_run=hparams.fast_dev_run,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        overfit_pct=hparams.overfit_pct,
        min_nb_epochs=-1, # Start early stopping immediately(it starts only AFTER min_nb_epochs done)
        weights_summary=None,
        nb_sanity_val_steps=(5 if sanity_check else 0)
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)
    best_val_results = get_best_val_result(trainer, logger)

    # ------------------------
    # 5 RESTORE BEST WEIGHTS AND TEST
    # ------------------------
    trainer.restore_weights(model)
    trainer.logger = None # Disable logging (would log on the same file as training)
    trainer.test()
    test_results = trainer.callback_metrics

    # Ensure everything is released
    del model
    del trainer

    return best_val_results, test_results

def get_best_val_result(trainer, logger):
    exp = logger.experiment
    best_epoch = trainer.early_stop_callback.stopped_epoch - trainer.early_stop_callback.wait
    log_per_epoch = len(exp.metrics) // (trainer.early_stop_callback.stopped_epoch + 1)
    tentative_best_res = exp.metrics[log_per_epoch * (best_epoch + 1) - 1]

    # HACK to deal with pytorch lightning bug
    if 'tn' not in tentative_best_res: # if this is not the evaluation step
        # when this bug occur, an extra training step is executed after the eval step
        tentative_best_res = exp.metrics[log_per_epoch * (best_epoch + 1) - 2]
        if 'tn' not in tentative_best_res:
            import pdb ; pdb.set_trace()

    return exp.metrics[log_per_epoch * (best_epoch + 1) - 1]
