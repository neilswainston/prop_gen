'''
(c) University of Liverpool 2019

All rights reserved.
'''
# pylint: disable=too-many-locals
# pylint: disable=wrong-import-order
from argparse import Namespace
from logging import Logger
import os

from typing import Tuple

from chemprop.utils import makedirs
import numpy as np
from .run_training import run_training


def cross_validate(args: Namespace, logger: Logger=None) \
        -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables:
    init_seed = args.seed
    save_dir = args.save_dir

    # Run training on different random seeds for each fold:
    all_scores = []

    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        all_scores.append(run_training(args, logger))

    _report(all_scores, args.data_df.columns, init_seed, args.metric, info)


def _report(all_scores, task_names, init_seed, metric, info):
    '''Report.'''
    # Report results:
    info(f'{len(all_scores)}-fold cross validation')

    all_scores = np.array(all_scores)

    # Report scores for each fold:
    for fold_num, scores in enumerate(all_scores):
        info(
            f'Seed {init_seed + fold_num} ==> test {metric} ='
            f' {np.nanmean(scores):.6f}')

        for task_name, score in zip(task_names, scores):
            info(
                f'Seed {init_seed + fold_num} ==> test {task_name}'
                f' {metric} = {score:.6f}')

    # Report scores across models:£
    # Average score for each model across tasks:
    avg_scores = np.nanmean(all_scores, axis=1)
    mean_score = np.nanmean(avg_scores)
    std_score = np.nanstd(avg_scores)

    info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')

    for task_num, task_name in enumerate(task_names):
        info(f'Overall test {task_name} {metric} = '
             f'{np.nanmean(all_scores[:, task_num]):.6f} +/-'
             f' {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score
