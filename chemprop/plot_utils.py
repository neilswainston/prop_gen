'''
(c) University of Liverpool 2019

All rights reserved.
'''
# pylint: disable=invalid-name
import matplotlib.pyplot as plt
import numpy as np


def plot(avg_test_preds, test_targets):
    '''Plot.'''
    plt.plot(avg_test_preds, test_targets, 'ro')

    x = np.linspace(-7, 3, 100)
    y = x
    plt.plot(x, y, '-g')
    plt.xlabel("Test Predictions")
    plt.ylabel("Test Targets")
    plt.title("Prediction Distribution")
    plt.savefig("Prediction_Distriution.png")
    plt.clf()

    x = np.linspace(-7, 3, 100)
    y = x - x
    plt.plot(x, y, '-g')
    plt.plot(test_targets, avg_test_preds - test_targets, 'rx')
    plt.xlabel("Test Targets")
    plt.ylabel("Test Errors")
    plt.title("Prediction Errors")
    plt.savefig("Prediction_Errors.png")
    plt.clf()
