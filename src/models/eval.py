"""Generic Evaluation methods modules"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from abc import ABC, abstractmethod
from typing import Dict, Iterable
from dataclasses import dataclass, field
from models.train import find_topn


def compute_mae(actual, predicted):
    """computes mean absoloute error between actual and prediction"""
    assert predicted.shape == actual.shape
    return mean_absolute_error(actual, predicted)


def compute_mean_recall(actual, predicted):
    """compute mean recall"""
    mean_recall = 0.0
    for index in range(actual.shape[0]):
        mean_recall = mean_recall + (
            len(np.intersect1d(actual[index],
                predicted[index])) / (len(actual[index]))
        )
    return mean_recall / actual.shape[0]


# TODO yet to be implemented
@dataclass
class EvalBase(ABC):
    """A simple base class that handles training code implementation
    Attributes
    ----------
    model_version : str
                    Unique indentifier for model training object

    X             : Iterable
                    Input data for training

    y             : Iterable, optional
                    Label of the data for training

    model_params  : Dict, optional
                    Configuration (parameters) of the model. Defaults to {}.

    data_config   : Dict, optional
                    Configuration about the data. Defaults to {}

    logger        : logger object
    """

    model_version: str
    X: Iterable
    y: Iterable = None
    model_params: Dict = field(default_factory=dict)
    data_config: Dict = field(default_factory=dict)
    logger: object = None
    kwargs: Dict = field(default_factory=dict)

    @abstractmethod
    def fit(self):
        pass

    def _fit(self):
        pass

    def save_model(self):
        pass
