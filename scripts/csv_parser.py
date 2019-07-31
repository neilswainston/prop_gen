'''
(c) University of Liverpool 2019

All rights reserved.
'''
import csv

from typing import List


def get_task_names(path: str, use_compound_names: bool=False) -> List[str]:
    """
    Gets the task names from a data CSV file.

    :param path: Path to a CSV file.
    :param use_compound_names: Whether file has compound names in addition to
    smiles strings.
    :return: A list of task names.
    """
    index = 2 if use_compound_names else 1

    return get_header(path)[index:]


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated
    header.
    """
    with open(path) as fle:
        header = next(csv.reader(fle))

    return header


def get_smiles(path: str, header: bool = True) -> List[str]:
    """
    Returns the smiles strings from a data CSV file (assuming the first line
    is a header).

    :param path: Path to a CSV file.
    :param header: Whether the CSV file contains a header (that will be
    skipped).
    :return: A list of smiles strings.
    """
    with open(path) as fle:
        reader = csv.reader(fle)
        if header:
            next(reader)  # Skip header
        smiles = [line[0] for line in reader]

    return smiles
