'''
(c) University of Liverpool 2019

All rights reserved.
'''
# pylint: disable=no-member
import argparse
import unittest

from rdkit import Chem

from chemprop.features.featurization import atom_features, bond_features, \
    get_atom_fdim, get_bond_fdim, onek_encoding_unk, MolGraph, ATOM_FEATURES


class Test(unittest.TestCase):
    '''Test class for featurization.'''

    def test_get_atom_fdim(self):
        '''Tests get_atom_fdim method.'''
        self.assertEqual(get_atom_fdim(), 133)

    def test_get_bond_fdim(self):
        '''Tests get_bond_fdim method.'''
        self.assertEqual(get_bond_fdim(), 14)

    def test_onek_encoding_unk(self):
        '''Tests onek_encoding_unk method.'''
        # Test value in choices:
        self.assertEqual(onek_encoding_unk(10, [1, 10, 100]),
                         [0, 1, 0, 0])

        # Test value not in choices:
        self.assertEqual(onek_encoding_unk(15, [1, 10, 100]),
                         [0, 0, 0, 1])

    def test_atom_features(self):
        '''Tests atom_features method.'''
        mol = Chem.MolFromSmiles('C')
        atm = mol.GetAtoms()[0]

        features = atom_features(atm)
        features[5] = 1

        # atomic_num:
        self.assertEqual(features[5], 1)

        # degree:
        self.assertEqual(
            features[len(ATOM_FEATURES['atomic_num']) + 1 + 4],
            1)

        # formal_charge:
        self.assertEqual(
            features[len(ATOM_FEATURES['atomic_num']) +
                     len(ATOM_FEATURES['degree']) + 2 + 1],
            1)

        # chiral_tag:
        self.assertEqual(
            features[len(ATOM_FEATURES['atomic_num']) +
                     len(ATOM_FEATURES['degree']) +
                     len(ATOM_FEATURES['formal_charge']) + 3 + 0],
            1)

        # num_Hs:
        self.assertEqual(
            features[len(ATOM_FEATURES['atomic_num']) +
                     len(ATOM_FEATURES['degree']) +
                     len(ATOM_FEATURES['formal_charge']) +
                     len(ATOM_FEATURES['chiral_tag']) + 4 + 4],
            1)

        # hybridization:
        self.assertEqual(
            features[len(ATOM_FEATURES['atomic_num']) +
                     len(ATOM_FEATURES['degree']) +
                     len(ATOM_FEATURES['formal_charge']) +
                     len(ATOM_FEATURES['chiral_tag']) +
                     len(ATOM_FEATURES['num_Hs']) + 5 + 2],
            1)

        # Is aromatic?:
        self.assertEqual(
            features[len(ATOM_FEATURES['atomic_num']) +
                     len(ATOM_FEATURES['degree']) +
                     len(ATOM_FEATURES['formal_charge']) +
                     len(ATOM_FEATURES['chiral_tag']) +
                     len(ATOM_FEATURES['num_Hs']) +
                     len(ATOM_FEATURES['hybridization']) + 6],
            0)

        # Mass:
        self.assertAlmostEqual(
            features[len(ATOM_FEATURES['atomic_num']) +
                     len(ATOM_FEATURES['degree']) +
                     len(ATOM_FEATURES['formal_charge']) +
                     len(ATOM_FEATURES['chiral_tag']) +
                     len(ATOM_FEATURES['num_Hs']) +
                     len(ATOM_FEATURES['hybridization']) + 7],
            0.12011)

    def test_bond_features(self):
        '''Tests bond_features method.'''
        mol = Chem.MolFromSmiles('C(=O)=O')
        bond = mol.GetBonds()[0]

        ftbond = bond_features(bond)

        # Is None?:
        self.assertEqual(ftbond[0], 0)

        # Single?:
        self.assertEqual(ftbond[1], False)

        # Double?:
        self.assertEqual(ftbond[2], True)

        # Triple?:
        self.assertEqual(ftbond[3], False)

        # Aromatic?:
        self.assertEqual(ftbond[4], False)

        # Conjugated?:
        self.assertEqual(ftbond[5], True)

        # In ring?:
        self.assertEqual(ftbond[6], False)

        # Get stereo?:
        self.assertEqual(ftbond[7:], [1, 0, 0, 0, 0, 0, 0])

    def test_mol_graph(self):
        '''Test MolGraph class.'''
        args = argparse.Namespace()
        args.atom_messages = False
        mol_graph = MolGraph('CCO', args)

        self.assertEqual(mol_graph.a2b, [[1], [0, 3], [2]])
        self.assertEqual(mol_graph.b2a, [0, 1, 1, 2])
        self.assertEqual(mol_graph.b2revb, [1, 0, 3, 2])


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
