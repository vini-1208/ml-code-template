"""Generic Training modules"""
import numpy as np


from abc import ABC, abstractmethod
from typing import Dict, Iterable
from dataclasses import dataclass, field


def find_topn(arr, top_n=20) -> np.ndarray:
    """keeps top N values in matrix and replaces remaining ones with 0."""
    arr_ = np.argsort(arr)
    arr[arr_[: arr_.shape[0] - top_n]] = 0
    return arr


@dataclass
class TrainingBase(ABC):
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
