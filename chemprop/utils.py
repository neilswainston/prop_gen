'''
(c) University of Liverpool 2019

All rights reserved.
'''
# pylint: disable=invalid-name
import logging
import os


def makedirs(path: str, isfile: bool = False):
    '''
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided
    (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    '''
    if isfile:
        path = os.path.dirname(path)

    if path:
        os.makedirs(path, exist_ok=True)


def create_logger(name: str, save_dir: str = None, quiet: bool = False) \
        -> logging.Logger:
    '''
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only
    saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only
    important info).
    :return: The logger.
    '''
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity:
    ch = logging.StreamHandler()

    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)

    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)

        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger
