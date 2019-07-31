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
import os
import pickle
import random
from rdkit import Chem
from typing import List, Tuple

from chemprop.features import load_features
import numpy as np
import pandas as pd

from .data import MoleculeDatapoint, MoleculeDataset
from .scaffold import log_scaffold_stats, scaffold_split


def get_data(skip_invalid_smiles: bool=True,
             args: Namespace=None,
             features_path: List[str]=None,
             max_data_size: int=None,
             use_compound_names: bool=None) -> MoleculeDataset:
    """
    Gets smiles string and target values (and optionally compound names if
    provided) from a CSV file.

    :param path: Path to a CSV file.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param args: Arguments.
    :param features_path: A list of paths to files containing features.
    If provided, it is used in place of args.features_path.
    :param max_data_size: The maximum number of data points to load.
    :param use_compound_names: Whether file has compound names in addition to
    smiles strings.
    :param logger: Logger.
    :return: A MoleculeDataset containing smiles strings and target values
    along with other info such as additional features and compound names when
    desired.
    """
    if args is not None:
        # Prefer explicit function arguments but default to args if not
        # provided
        features_path = features_path if features_path else args.features_path
        max_data_size = max_data_size if max_data_size else args.max_data_size
        use_compound_names = use_compound_names if use_compound_names \
            else args.use_compound_names
    else:
        use_compound_names = False

    max_data_size = max_data_size or float('inf')

    # Load features
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            # each is num_data x num_features
            features_data.append(load_features(feat_path))
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    # Load data
    data = []

    for idx, (smiles, targets) in \
            enumerate(args.data_df.head(min(max_data_size,
                                            len(args.data_df))).iterrows()):

        if smiles:
            mol = Chem.MolFromSmiles(smiles)

        if not skip_invalid_smiles or (mol and mol.GetNumHeavyAtoms()):
            data.append(MoleculeDatapoint(
                smiles=smiles,
                mol=mol,
                targets=targets,
                features=features_data[idx] if features_data else None,
                features_generator=args.features_generator,
                compound_name=None
            ))

    return MoleculeDataset(data)


def get_data_from_smiles(smiles: List[str], skip_invalid_smiles: bool=True) \
        -> MoleculeDataset:
    """
    Converts SMILES to a MoleculeDataset.

    :param smiles: A list of SMILES strings.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param logger: Logger.
    :return: A MoleculeDataset with all of the provided SMILES.
    """
    data = []

    for smile in smiles:
        if smile:
            mol = Chem.MolFromSmiles(smiles)

            if not skip_invalid_smiles or (mol and mol.GetNumHeavyAtoms()):
                data.append((smile, mol))

    return MoleculeDataset([MoleculeDatapoint(d[0], d[1], None)
                            for d in data])


def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               args: Namespace = None,
               logger: Logger = None) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
    """
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
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    if args is not None:
        folds_file, val_fold_index, test_fold_index = \
            args.folds_file, args.val_fold_index, args.test_fold_index
    else:
        folds_file = val_fold_index = test_fold_index = None

    if split_type == 'crossval':
        index_set = args.crossval_index_sets[args.seed]
        data_split = []
        for split in range(3):
            split_indices = []
            for index in index_set[split]:
                with open(os.path.join(
                        args.crossval_index_dir, f'{index}.pkl'), 'rb') as rf:
                    split_indices.extend(pickle.load(rf))
            data_split.append([data[i] for i in split_indices])
        train, val, test = tuple(data_split)

        return MoleculeDataset(train), MoleculeDataset(val), \
            MoleculeDataset(test)

    if split_type == 'index_predetermined':
        split_indices = args.crossval_index_sets[args.seed]
        assert len(split_indices) == 3
        data_split = []
        for split in range(3):
            data_split.append([data[i] for i in split_indices[split]])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), \
            MoleculeDataset(test)

    if split_type == 'predetermined':
        if not val_fold_index:
            # test set is created separately so use all of the other data for
            # train and val
            assert sizes[2] == 0
        assert folds_file is not None
        assert test_fold_index is not None

        try:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f)
        except UnicodeDecodeError:
            with open(folds_file, 'rb') as f:
                # in case we're loading indices from python2
                all_fold_indices = pickle.load(f, encoding='latin1')
        # assert len(data) == \
        # sum([len(fold_indices) for fold_indices in all_fold_indices])

        log_scaffold_stats(data, all_fold_indices, logger=logger)

        folds = [[data[i] for i in fold_indices]
                 for fold_indices in all_fold_indices]

        test = folds[test_fold_index]
        if val_fold_index is not None:
            val = folds[val_fold_index]

        train_val = []
        for i, fold in enumerate(folds):
            if i != test_fold_index and \
                    (not val_fold_index or i != val_fold_index):
                train_val.extend(fold)

        if val_fold_index is not None:
            train = train_val
        else:
            random.seed(seed)
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

        return MoleculeDataset(train), MoleculeDataset(val), \
            MoleculeDataset(test)

    if split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, seed=seed,
                              logger=logger)

    if split_type == 'random':
        data.shuffle(seed=seed)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]

        return MoleculeDataset(train), MoleculeDataset(val), \
            MoleculeDataset(test)

    raise ValueError(f'split_type "{split_type}" not supported.')


def get_class_sizes(data_df: pd.DataFrame) -> List[List[float]]:
    """
    Determines the proportions of the different classes in the classification
    dataset.

    :param data: A Pandas DataFrame
    :return: A list of lists of class proportions. Each inner list contains the
    class proportions
    for a task.
    """
    for col in data_df.columns:
        assert data_df[col].dropna().isin([0, 1]).all()

    return pd.DataFrame([data_df[col].value_counts(normalize=True)
                         for col in data_df.columns])
