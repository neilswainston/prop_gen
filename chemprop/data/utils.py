'''
(c) University of Liverpool 2019

All rights reserved.
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
from argparse import Namespace
from logging import Logger
from typing import List, Tuple

from rdkit import Chem
from sklearn.preprocessing.data import StandardScaler

from chemprop.features import load_features
import numpy as np
import pandas as pd

from .data import MoleculeDatapoint, MoleculeDataset
from .scaffold import scaffold_split


def get_data(args, logger, debug):
    '''Get data.'''
    # Get data:
    train_data, val_data, test_data = _get_data(args, logger)

    debug(f'train size = {len(train_data):,} | val size = {len(val_data):,} |'
          f' test size = {len(test_data):,}')

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(args.data_df)
        debug('Class sizes')
        debug(class_sizes)

    # Scale features:
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
        train_data.set_targets(targets)
    else:
        scaler = None

    return train_data, val_data, test_data, scaler, features_scaler


def get_data_from_smiles(smiles: List[str], skip_invalid_smiles: bool=True) \
        -> MoleculeDataset:
    '''
    Converts SMILES to a MoleculeDataset.

    :param smiles: A list of SMILES strings.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param logger: Logger.
    :return: A MoleculeDataset with all of the provided SMILES.
    '''
    data = []

    for smile in smiles:
        if smile:
            mol = Chem.MolFromSmiles(smiles)

            if not skip_invalid_smiles or (mol and mol.GetNumHeavyAtoms()):
                data.append((smile, mol))

    return MoleculeDataset([MoleculeDatapoint(d[0], d[1], None)
                            for d in data])


def get_class_sizes(data_df: pd.DataFrame) -> List[List[float]]:
    '''
    Determines the proportions of the different classes in the classification
    dataset.

    :param data: A Pandas DataFrame
    :return: A list of lists of class proportions. Each inner list contains the
    class proportions
    for a task.
    '''
    for col in data_df.columns:
        assert data_df[col].dropna().isin([0, 1]).all()

    return pd.DataFrame([data_df[col].value_counts(normalize=True)
                         for col in data_df.columns])


def _get_data(args: Namespace, logger: Logger = None):
    '''Get data.'''
    if logger:
        debug = logger.debug
    else:
        debug = print

    debug('Loading data')
    args.task_names = args.data_df.columns
    data = _get_data_from_df(data_df=args.data_df)
    args.num_tasks = data.num_tasks()
    args.features_size = len(args.data_df.columns)
    debug(f'Number of tasks = {args.num_tasks}')

    if args.separate_test_path:
        test_data = _get_data_from_df(
            data_df=pd.read_csv(args.separate_test_path),
            features_path=args.separate_test_features_path)
    else:
        test_data = None

    if args.separate_val_path:
        val_data = _get_data_from_df(
            data_df=pd.read_csv(args.separate_val_path),
            features_path=args.separate_val_features_path)
    else:
        val_data = None

    if args.separate_val_path and args.separate_test_path:
        return data, val_data, test_data

    # Split data
    debug(f'Splitting data with seed {args.seed}')

    if val_data:
        train_data, _, test_data = _split_data(data=data,
                                               split_type=args.split_type,
                                               sizes=(0.8, 0.2, 0.0),
                                               seed=args.seed,
                                               logger=logger)

        return train_data, val_data, test_data

    if test_data:
        train_data, val_data, _ = _split_data(data=data,
                                              split_type=args.split_type,
                                              sizes=(0.8, 0.2, 0.0),
                                              seed=args.seed,
                                              logger=logger)

        return train_data, val_data, test_data

    return _split_data(data=data,
                       split_type=args.split_type,
                       sizes=args.split_sizes,
                       seed=args.seed,
                       logger=logger)


def _get_data_from_df(data_df: pd.DataFrame,
                      skip_invalid_smiles: bool=True,
                      features_path: List[str]=None,
                      features_generator: List[str] = None,
                      max_data_size: int=None) -> MoleculeDataset:
    '''
    Gets smiles string and target values (and optionally compound names if
    provided) from a CSV file.

    :param data_df: DataFrame from parsed CSV file.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param args: Arguments.
    :param features_path: A list of paths to files containing features.
    If provided, it is used in place of args.features_path.
    :param features_generator: List of strings of features_generator names.
    :param max_data_size: The maximum number of data points to load.
    :return: A MoleculeDataset containing smiles strings and target values
    along with other info such as additional features and compound names when
    desired.
    '''
    max_data_size = max_data_size or float('inf')

    # Load features:
    if features_path:
        features_data = []

        for feat_path in features_path:
            # Each is num_data x num_features:
            features_data.append(load_features(feat_path))

        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    # Load data
    data = []

    for idx, (smiles, targets) in \
            enumerate(data_df.head(min(max_data_size,
                                       len(data_df))).iterrows()):

        if smiles:
            mol = Chem.MolFromSmiles(smiles)

        if not skip_invalid_smiles or (mol and mol.GetNumHeavyAtoms()):
            data.append(MoleculeDatapoint(
                smiles=smiles,
                mol=mol,
                targets=targets,
                features=features_data[idx] if features_data else None,
                features_generator=features_generator,
                compound_name=None
            ))

    return MoleculeDataset(data)


def _split_data(data: MoleculeDataset,
                split_type: str = 'random',
                sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                seed: int = 0,
                logger: Logger = None) -> Tuple[MoleculeDataset,
                                                MoleculeDataset,
                                                MoleculeDataset]:
    '''
    Splits data into training, validation, and test splits.

    :param data: A MoleculeDataset.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param seed: The random seed to use before shuffling data.
    :param args: Namespace of arguments.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the
    data.
    '''
    assert len(sizes) == 3 and sum(sizes) == 1

    if split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, seed=seed,
                              logger=logger)

    if split_type == 'random':
        return _split_random(data, sizes, seed)

    raise ValueError(f'split_type {split_type} not supported.')


def _split_random(data: MoleculeDataset,
                  sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                  seed: int = 0):
    '''Split random.'''
    data.shuffle(seed=seed)

    train_size = int(sizes[0] * len(data))
    val_size = int((sizes[0] + sizes[1]) * len(data))

    train = data[:train_size]
    val = data[train_size:val_size]
    test = data[val_size:]

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)
