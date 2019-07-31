'''
(c) University of Liverpool 2019

All rights reserved.
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=wrong-import-order
import random

from rdkit import Chem
from torch.utils.data.dataset import Dataset
from typing import Callable, List, Union

from chemprop.features import get_features_generator
import numpy as np

from .scaler import StandardScaler


class MoleculeDatapoint:
    '''
    A MoleculeDatapoint contains a single molecule and its associated features
    and targets.
    '''

    def __init__(self,
                 smiles: str,
                 mol: Chem.Mol,
                 targets: np.ndarray,
                 features: np.ndarray = None,
                 features_generator: List[str] = None,
                 compound_name: str = None):
        '''
        Initializes a MoleculeDatapoint, which contains a single molecule.

        :param smiles: Smiles string
        :param mol: Chem.Mol
        :param targets: A numpy array containing features
        :param features: A numpy array containing additional features
        (e.g. ƒMorgan fingerprint).
        :param features_generator: List of strings of features_generator names.
        :param use_compound_names: Whether the data CSV includes the compound
        name on each line.
        '''
        self.smiles = smiles
        self.mol = mol
        self.compound_name = compound_name

        if features and features_generator:
            raise ValueError('Currently cannot provide both loaded features'
                             ' and a features generator.')

        if features_generator:
            # Generate additional features if given a generator:
            features = []

            for fg in features_generator:
                features_generator = get_features_generator(fg)

                if self.mol and self.mol.GetNumHeavyAtoms():
                    self.features.extend(features_generator(self.mol))

            features = np.array(self.features)

        self.set_features(features)
        self.set_targets(targets)

    def set_features(self, features: np.ndarray):
        '''
        Sets the features of the molecule.

        :param features: A 1-D numpy array of features for the molecule.
        '''
        self.features = features

        # Fix NaNs in features:
        if self.features:
            self.features = np.where(np.isnan(self.features), 0, self.features)

    def set_targets(self, targets: List[float]):
        '''
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        '''
        self.targets = targets


class MoleculeDataset(Dataset):
    '''
    A MoleculeDataset contains a list of molecules and their associated
    features and targets.
    '''

    def __init__(self, data: List[MoleculeDatapoint]):
        '''
        Initializes a MoleculeDataset, which contains a list of
        MoleculeDatapoints (i.e. a list of molecules).

        :param data: A list of MoleculeDatapoints.
        '''
        self.data = data

    def compound_names(self) -> List[str]:
        '''
        Returns the compound names associated with the molecule (if they
        exist).

        :return: A list of compound names or None if the dataset does not
        contain compound names.
        '''
        if not self.data or not self.data[0].compound_name:
            return None

        return [d.compound_name for d in self.data]

    def smiles(self) -> List[str]:
        '''
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        '''
        return [d.smiles for d in self.data]

    def mols(self) -> List[Chem.Mol]:
        '''
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        '''
        return [d.mol for d in self.data]

    def features(self) -> List[np.ndarray]:
        '''
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each
        molecule or None if there are no features.
        '''
        if not self.data or not self.data[0].features:
            return None

        return [d.features for d in self.data]

    def targets(self) -> List[List[float]]:
        '''
        Returns the targets associated with each molecule.

        :return: A list of lists of floats containing the targets.
        '''
        return [d.targets for d in self.data]

    def num_tasks(self) -> int:
        '''
        Returns the number of prediction tasks.

        :return: The number of tasks.
        '''
        return len(self.data[0].targets) if self.data else None

    def features_size(self) -> int:
        '''
        Returns the size of the features array associated with each molecule.

        :return: The size of the features.
        '''
        return len(self.data[0].features) \
            if self.data and self.data[0].features \
            else None

    def shuffle(self, seed: int=None):
        '''
        Shuffles the dataset.

        :param seed: Optional random seed.
        '''
        if seed:
            random.seed(seed)

        random.shuffle(self.data)

    def normalize_features(self, scaler: StandardScaler=None,
                           replace_nan_token: int=0) -> StandardScaler:
        '''
        Normalizes the features of the dataset using a StandardScaler
        (subtract mean, divide by standard deviation).

        If a scaler is provided, uses that scaler to perform the normalization.
        Otherwise fits a scaler to the features in the dataset and then
        performs the normalization.

        :param scaler: A fitted StandardScaler. Used if provided.
        Otherwise a StandardScaler is fit on this dataset and is then used.
        :param replace_nan_token: What to replace nans with.
        :return: A fitted StandardScaler. If a scaler is provided, this is the
        same scaler. Otherwise, this is a scaler fit on this dataset.
        '''
        if not self.data or not self.data[0].features:
            return None

        if not scaler:
            scaler = StandardScaler(replace_nan_token=replace_nan_token)

        features = np.vstack([d.features for d in self.data])
        scaler.fit(features)

        for d in self.data:
            d.set_features(scaler.transform(d.features.reshape(1, -1))[0])

        return scaler

    def set_targets(self, targets: List[List[float]]):
        '''
        Sets the targets for each molecule in the dataset. Assumes the targets
        are aligned with the datapoints.

        :param targets: A list of lists of floats containing targets for each
        molecule. This must be the same length as the underlying dataset.
        '''
        assert len(self.data) == len(targets)

        for i, d in enumerate(self.data):
            d.set_targets(targets[i])

    def sort(self, key: Callable):
        '''
        Sorts the dataset using the provided key.

        :param key: A function on a MoleculeDatapoint to determine the sorting
        order.
        '''
        self.data.sort(key=key)

    def __len__(self) -> int:
        '''
        Returns the length of the dataset (i.e. the number of molecules).

        :return: The length of the dataset.
        '''
        return len(self.data)

    def __getitem__(self, item) \
            -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        '''
        Gets one or more MoleculeDatapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A MoleculeDatapoint if an int is provided or a list of
        MoleculeDatapoints if a slice is provided.
        '''
        return self.data[item]
