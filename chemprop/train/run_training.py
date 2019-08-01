'''
(c) University of Liverpool 2019

All rights reserved.
'''
# pylint: disable=fixme
# pylint: disable=invalid-name
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
from argparse import Namespace
from logging import Logger
import os
from pprint import pformat

from sklearn.preprocessing.data import StandardScaler
from tensorboardX import SummaryWriter
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange
from typing import List

from chemprop.data.utils import get_class_sizes, get_data
from chemprop.models import build_model
from chemprop.models.utils import build_lr_scheduler, \
    get_loss_func, get_metric_func, load_checkpoint, save_checkpoint
from chemprop.nn_utils import param_count
from chemprop.plot_utils import plot
from chemprop.utils import makedirs
import numpy as np

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train


def run_training(args: Namespace, logger: Logger = None) -> List[float]:
    '''
    Trains a model and returns test scores on the model checkpoint with the
    highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    '''
    if logger:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Print args:
    debug(pformat(vars(args)))

    # Set GPU
    if args.gpu:
        torch.cuda.set_device(args.gpu)

    # Get data:
    train_data, val_data, test_data = get_data(args, logger)

    debug(f'train size = {len(train_data):,} | val size = {len(val_data):,} |'
          f' test size = {len(test_data):,}')

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(args.data_df)
        debug('Class sizes')
        debug(class_sizes)

    if args.features_scaling:
        features_scaler = train_data.normalize_features()
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    # Initialise scaler and scale training targets by subtracting mean and
    # dividing standard deviation (regression only):
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        scaler = StandardScaler()
        targets = scaler.fit_transform(train_data.targets())
        train_data.smiles().set_targets(targets.tolist())
    else:
        scaler = None

    # Set up test set evaluation:
    test_targets = test_data.targets()

    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_data.smiles()),
                                   args.num_tasks,
                                   args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_data.smiles()), args.num_tasks))

    # Setup val set evaluation:
    if args.dataset_type == 'multiclass':
        sum_val_preds = np.zeros((len(val_data.smiles()),
                                  args.num_tasks,
                                  args.multiclass_num_classes))
    else:
        sum_val_preds = np.zeros((len(val_data.smiles()), args.num_tasks))

    # Train ensemble of models:
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer:
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        writer = SummaryWriter(logdir=save_dir)

        # Load/build model:
        if args.checkpoint_paths:
            debug(
                f'Loading model {model_idx} from'
                f' {args.checkpoint_paths[model_idx]}')

            model = load_checkpoint(args.checkpoint_paths[model_idx],
                                    current_args=args,
                                    logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')

        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0
        # epochs:
        save_checkpoint(os.path.join(save_dir, 'model.pt'),
                        model, scaler, features_scaler, args)

        # Optimizers:
        optimizer = Adam([{'params': model.parameters(),
                           'lr': args.init_lr,
                           'weight_decay': 0}])

        # Learning rate schedulers:
        scheduler = build_lr_scheduler(optimizer,
                                       warmup_epochs=args.warmup_epochs,
                                       train_data_size=len(train_data),
                                       batch_size=args.batch_size,
                                       init_lr=args.init_lr,
                                       max_lr=args.max_lr,
                                       epochs=args.epochs,
                                       num_lrs=args.num_lrs,
                                       final_lr=args.final_lr)

        # Run training:
        best_score = float('inf') if args.minimize_score else -float('inf')

        best_epoch = 0
        n_iter = 0

        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                data=train_data,
                loss_func=get_loss_func(args.dataset_type),
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )

            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            val_scores = evaluate(
                model=model,
                data=val_data,
                num_tasks=args.num_tasks,
                metric_func=get_metric_func(metric=args.metric),
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )

            # Average validation score:
            avg_val_score = np.nanmean(val_scores)
            debug(f'Validation {args.metric} = {avg_val_score:.6f}')

            writer.add_scalar(
                f'validation_{args.metric}', avg_val_score, n_iter)

            if args.show_individual_scores:
                # Individual validation scores:
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(
                        f'Validation {task_name} {args.metric} ='
                        f' {val_score:.6f}')

                    writer.add_scalar(
                        f'validation_{task_name}_{args.metric}', val_score,
                        n_iter)

            # Save model checkpoint if improved validation score:
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch

                save_checkpoint(os.path.join(save_dir, 'model.pt'),
                                model, scaler, features_scaler, args)

        # Evaluate on test set using model with best validation score:
        info(
            f'Model {model_idx} best validation {args.metric} ='
            f' {best_score:.6f} on epoch {best_epoch}')

        model = load_checkpoint(os.path.join(
            save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

        # todo: Change code here to analyze the model on the trained data.

        val_preds = predict(
            model=model,
            data=val_data,
            batch_size=args.batch_size,
            scaler=scaler
        )

        if val_preds:
            sum_val_preds += np.array(val_preds)

        test_preds = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )

        if test_preds:
            sum_test_preds += np.array(test_preds)

        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=get_metric_func(metric=args.metric),
            dataset_type=args.dataset_type,
            logger=logger
        )

        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
        writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(
                    f'Model {model_idx} test {task_name} {args.metric} ='
                    f' {test_score:.6f}')
                writer.add_scalar(
                    f'test_{task_name}_{args.metric}', test_score, n_iter)

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
    avg_val_preds = (sum_val_preds / args.ensemble_size).tolist()

    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=get_metric_func(metric=args.metric),
        dataset_type=args.dataset_type,
        logger=logger
    )

    print('Test Prediction Shape:- ', np.array(avg_test_preds).shape)

    avg_test_preds = np.array(avg_test_preds).reshape(1, -1)
    test_targets = np.array(test_targets).reshape(1, -1)
    avg_val_preds = np.array(avg_val_preds).reshape(1, -1)
    # val_targets = np.array(test_targets).reshape(1, -1)

    smaller_count = np.sum(avg_test_preds < test_targets)
    smaller_frac = smaller_count / (avg_test_preds.shape[1])
    print('Smaller_Fraction: ', smaller_frac)

    # Plot:
    plot(avg_test_preds, test_targets)

    # Average ensemble score
    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
    writer.add_scalar(
        f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)

    # Individual ensemble scores
    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            info(
                f'Ensemble test {task_name} {args.metric} ='
                f' {ensemble_score:.6f}')

    return ensemble_scores
