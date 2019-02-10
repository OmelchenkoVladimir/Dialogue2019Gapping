import numpy as np
import pandas as pd

from base_solution import BaseSolution
from sklearn.base import BaseEstimator


class BinaryHyphenBaselineSolution(BaseEstimator):
    def __init__(self):
        pass;

    def fit(self, train):
        pass;

    def predict(self, valid):
        res = pd.DataFrame(valid['text'].apply(lambda x: 1 if x.count('-') > 0 else 0))
        res.columns = ['class']
        return res


class BinaryHyphenBaseline(BaseSolution):
    def create_solutions(self):
        sol = BinaryHyphenBaselineSolution()
        return [(sol.get_params(), sol)]
