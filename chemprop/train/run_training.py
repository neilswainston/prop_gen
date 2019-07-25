'''
(c) University of Liverpool 2019

All rights reserved.
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
from argparse import Namespace
import csv
from logging import Logger
import os
import pickle
from pprint import pformat

from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange
from typing import List

from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, split_data
from chemprop.models import build_model
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, \
    get_loss_func, get_metric_func, load_checkpoint, makedirs, save_checkpoint
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train


# Matplotlib
def run_training(args: Namespace, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the
    highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu:
        torch.cuda.set_device(args.gpu)

    # Print args
    debug(pformat(vars(args)))

    # Get data
    debug('Loading data')
    args.task_names = args.data_df.columns
    data = get_data(args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = len(args.data_df.columns)
    debug(f'Number of tasks = {args.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args,
                             features_path=args.separate_test_features_path,
                             logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args,
                            features_path=args.separate_val_features_path,
                            logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data,
                                              split_type=args.split_type,
                                              sizes=(0.8, 0.2, 0.0),
                                              seed=args.seed,
                                              args=args,
                                              logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data,
                                             split_type=args.split_type,
                                             sizes=(0.8, 0.2, 0.0),
                                             seed=args.seed,
                                             args=args,
                                             logger=logger)
    else:
        train_data, val_data, test_data = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            seed=args.seed,
            args=args,
            logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(args.data_df)
        debug('Class sizes')
        debug(class_sizes)

    if args.save_smiles_splits:
        with open(args.data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            lines_by_smiles = {}
            indices_by_smiles = {}
            for i, line in enumerate(reader):
                smiles = line[0]
                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i

        all_split_indices = []
        for dataset, name in [(train_data, 'train'), (val_data, 'val'),
                              (test_data, 'test')]:
            with open(os.path.join(
                    args.save_dir, name + '_smiles.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['smiles'])
                for smiles in dataset.smiles():
                    writer.writerow([smiles])
            with open(os.path.join(
                    args.save_dir, name + '_full.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for smiles in dataset.smiles():
                    writer.writerow(lines_by_smiles[smiles])
            split_indices = []
            for smiles in dataset.smiles():
                split_indices.append(indices_by_smiles[smiles])
                split_indices = sorted(split_indices)
            all_split_indices.append(split_indices)
        with open(os.path.join(
                args.save_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} |'
          f' test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and
    # dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros(
            (len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Setup val set evaluation
    val_smiles, _ = val_data.smiles(), val_data.targets()
    if args.dataset_type == 'multiclass':
        sum_val_preds = np.zeros(
            (len(val_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_val_preds = np.zeros((len(val_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(
                f'Loading model {model_idx} from'
                f' {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(
                args.checkpoint_paths[model_idx], current_args=args,
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
        # epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'),
                        model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                data=train_data,
                loss_func=loss_func,
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
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )

            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            debug(f'Validation {args.metric} = {avg_val_score:.6f}')
            writer.add_scalar(
                f'validation_{args.metric}', avg_val_score, n_iter)

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(
                        f'Validation {task_name} {args.metric} ='
                        f' {val_score:.6f}')
                    writer.add_scalar(
                        f'validation_{task_name}_{args.metric}', val_score,
                        n_iter)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'),
                                model, scaler, features_scaler, args)

        # Evaluate on test set using model with best validation score
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

        test_preds = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        if val_preds:
            sum_val_preds += np.array(val_preds)

        if test_preds:
            sum_test_preds += np.array(test_preds)

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
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    print("Test Prediction Shape:- ", np.array(avg_test_preds).shape)

    avg_test_preds = np.array(avg_test_preds).reshape(1, -1)
    test_targets = np.array(test_targets).reshape(1, -1)
    avg_val_preds = np.array(avg_val_preds).reshape(1, -1)
    # val_targets = np.array(test_targets).reshape(1, -1)

    smaller_count = np.sum(avg_test_preds < test_targets)
    smaller_frac = smaller_count / (avg_test_preds.shape[1])
    print("Smaller_Fraction: ", smaller_frac)

    # plt.plot(np.concatenate((avg_test_preds, avg_val_preds), axis=1),
    #         np.concatenate((test_targets, val_targets), axis=1), 'rx')
    plt.plot(avg_test_preds, test_targets, 'ro')
    # x = np.linspace(0, 11000, 110000)
    x = np.linspace(-7, 3, 100)
    y = x
    plt.plot(x, y, '-g')
    plt.xlabel("Test Predictions")
    plt.ylabel("Test Targets")
    plt.title("Prediction Distribution")
    plt.savefig("Prediction_Distriution.png")
    # plt.show()
    plt.clf()
    x = np.linspace(-7, 3, 100)
    y = x - x
    plt.plot(x, y, '-g')
    plt.plot(test_targets, avg_test_preds - test_targets, 'rx')
    plt.xlabel("Test Targets")
    plt.ylabel("Test Errors")
    plt.title("Prediction Errors")
    plt.savefig("Prediction_Errors.png")
    # plt.show()
    plt.clf()

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
