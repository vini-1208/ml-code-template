"""Generic Prediction modules"""
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Dict, Iterable
from dataclasses import dataclass

from constants.model_constants import RATING_METHOD


@dataclass
class PredictionBase(ABC):
    """A simple base class that handles prediction code implementation
    Attributes
    ----------
    model_version : str
                    Unique indentifier for model training object

    X             : Iterable
                    Input data for prediction (can be pandas, numpy or tensor object)

    model_obj     : Model
                    Serialized model object. Defaults to None

    logger        : logger object
    """

    model_version: str
    X: Iterable
    model_obj: object = None
    logger: object = None
    kwargs: Dict = None

    @abstractmethod
    def predict(self):
        pass


class CFPrediction(PredictionBase):
    """Collaborative Filtering implementation for generating predictions"""

    def predict(self, ):
        """this function computes a prediction matrix of user ratings.
        Takes two inputs (initiated as class attributes) for computation: i) User Rating Matrix ii) Similarity Matrix

        Raises
        ------
        ValueError
           If rating_method is invalid


        Returns
        -------
        prediction_mat : numpy
           Prediction matrix computed

        prediction_eval_mat : numpy
            Prediction matrix for evaluating against actual rating matrix

        prediction_ranking_mat : numpy
           Prediction matrix to make recommendations based on ranking
        """

        return (prediction_mat, prediction_eval_mat, prediction_ranking_mat)
