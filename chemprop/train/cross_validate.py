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

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = args.data_df.columns

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores = run_training(args, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(
            f'Seed {init_seed + fold_num} ==> test {args.metric} ='
            f' {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(
                    f'Seed {init_seed + fold_num} ==> test {task_name}'
                    f' {args.metric} = {score:.6f}')

    # Report scores across models
    # average score for each model across tasks
    avg_scores = np.nanmean(all_scores, axis=1)
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/-'
                 f' {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score
