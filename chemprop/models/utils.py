'''
(c) University of Liverpool 2019

All rights reserved.
'''
# pylint: disable=invalid-name
# pylint: disable=fixme
# pylint: disable=no-member
# pylint: disable=too-many-arguments
# pylint: disable=too-many-return-statements
# pylint: disable=ungrouped-imports
# pylint: disable=wrong-import-order
from argparse import Namespace
import logging
import math
from typing import Callable, List, Tuple, Union

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, \
    precision_recall_curve, r2_score, roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing.data import StandardScaler
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from chemprop.models import build_model, MoleculeModel
from chemprop.nn_utils import NoamLR
import numpy as np
from robust_loss_pytorch import adaptive
import torch.nn as nn


def save_checkpoint(path: str,
                    model: MoleculeModel,
                    scaler: StandardScaler = None,
                    features_scaler: StandardScaler = None,
                    args: Namespace = None):
    '''
    Saves a model checkpoint.

    :param model: A MoleculeModel.
    :param scaler: A StandardScaler fitted on the data.
    :param features_scaler: A StandardScaler fitted on the features.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    '''
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': scaler.mean_,
            'stds': scaler.var_**0.5
        } if scaler else None,
        'features_scaler': {
            'means': features_scaler.mean_,
            'stds': features_scaler.scaler.var_**0.5
        } if features_scaler else None
    }
    torch.save(state, path)


def load_checkpoint(path: str,
                    current_args: Namespace = None,
                    cuda: bool = None,
                    logger: logging.Logger = None) -> MoleculeModel:
    '''
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded
    from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    '''
    debug = logger.debug if logger else print

    # Load model and args:
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']

    if current_args is not None:
        args = current_args

    args.cuda = cuda if cuda else args.cuda

    # Build model
    model = build_model(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        if param_name not in model_state_dict:
            debug(
                f'Pretrained parameter {param_name} cannot be found in model'
                ' parameters.')
        elif model_state_dict[param_name].shape != \
                loaded_state_dict[param_name].shape:
            debug(f'Pretrained parameter {param_name} '
                  f'of shape {loaded_state_dict[param_name].shape} does not'
                  ' match corresponding model parameter of shape'
                  f' {model_state_dict[param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter {param_name}.')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights:
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    return model


def load_scalers(path: str) -> Tuple[StandardScaler, StandardScaler]:
    '''
    Loads the scalers a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data scaler and the features scaler.
    '''
    state = torch.load(path, map_location=lambda storage, loc: storage)

    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) \
        if state['data_scaler'] else None

    features_scaler = StandardScaler(state['features_scaler']['means'],
                                     state['features_scaler']['stds']) \
        if state['features_scaler'] else None

    return scaler, features_scaler


def load_args(path: str) -> Namespace:
    '''
    Loads the arguments a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The arguments Namespace that the model was trained with.
    '''
    return torch.load(path, map_location=lambda storage, loc: storage)['args']


def get_loss_func(dataset_type: str) -> nn.Module:
    '''
    Gets the loss function corresponding to a given dataset type.

    :param dataset_type: str containing the dataset type
    ('classification' or 'regression').
    :return: A PyTorch loss function.
    '''
    if dataset_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')

    if dataset_type == 'regression':
        return nn.MSELoss(reduction='none')

    if dataset_type == 'multiclass':
        return nn.CrossEntropyLoss(reduction='none')

    # todo: change the loss function here:
    if dataset_type == 'dopamine':
        # return quantile_loss_func(0.75)
        # return quantile_loss_func(0.7)
        return adaptive_loss_func()
        # return simple_heteroscedastic_loss_func()

    raise ValueError(f'Dataset type {dataset_type} not supported.')


def quantile_loss_func(alpha):
    '''quantile_loss_func.'''

    def loss_func(preds, targets):
        d = (preds - targets)
        return ((d ** 2) * (alpha + torch.sign(d)) ** 2).mean()

    return loss_func


def adaptive_loss_func():
    '''adaptive_loss_func.'''

    def loss_func(preds, targets):
        adaptive_lossfun = adaptive.AdaptiveLossFunction(1, np.float32, 'cuda')
        d = torch.as_tensor(preds - targets)
        loss = torch.sum(adaptive_lossfun.lossfun(d))
        return loss

    return loss_func


def simple_heteroscedastic_loss_func():
    '''simple_heteroscedastic_loss_func.'''

    def loss_func(preds, targets):
        w = ((100.0 - 1.0) / (0.059 - 1000000.0)) * (targets - 1.0) + 1.0
        ls = ((preds - targets)**2) * w
        return torch.mean(ls)

    return loss_func


def prc_auc(targets: List[int], preds: List[float]) -> float:
    '''
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    '''
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def rmse(targets: List[float], preds: List[float]) -> float:
    '''
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    '''
    return math.sqrt(mean_squared_error(targets, preds))


def accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) \
        -> float:
    '''
    Computes the accuracy of a binary prediction task using a given threshold
    for generating hard predictions.
    Alternatively, compute accuracy for a multiclass prediction task by picking
    the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below
    which (inclusive) a prediction is a 0
    :return: The computed accuracy.
    '''
    if isinstance(preds[0], list):  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        # binary prediction
        hard_preds = [1 if p > threshold else 0 for p in preds]
    return accuracy_score(targets, hard_preds)


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]],
                                              List[float]], float]:
    '''
    Gets the metric function corresponding to a given metric name.

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a
    list of predictions and returns.
    '''
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    if metric == 'accuracy':
        return accuracy

    if metric == 'cross_entropy':
        return log_loss

    raise ValueError(f'Metric {metric} not supported.')


def build_lr_scheduler(optimizer: Optimizer,
                       warmup_epochs: int,
                       train_data_size: int,
                       batch_size: int,
                       init_lr: float,
                       max_lr: float,
                       final_lr: float,
                       epochs: int,
                       num_lrs: int,
                       total_epochs: List[int] = None) -> _LRScheduler:
    '''
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param total_epochs: The total number of epochs for which the model will be
    run.
    :return: An initialized learning rate scheduler.
    '''
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[warmup_epochs],
        total_epochs=total_epochs or [epochs] * num_lrs,
        steps_per_epoch=train_data_size // batch_size,
        init_lr=[init_lr],
        max_lr=[max_lr],
        final_lr=[final_lr]
    )
