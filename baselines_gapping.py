import numpy as np
import pandas as pd

from base_solution import BaseSolution
from sklearn.base import BaseEstimator
from extra import find_last_verb_position


class GappingHyphenBaselineSolution(BaseEstimator):
    def __init__(self):
        pass;

    def fit(self, train):
        pass;

    def predict(self, valid):
        def lambda_cv(row):
            if row['V'] == '-1:-1':
                return '-1:-1'
            a, b = find_last_verb_position(row['text'][:int(row['V'].split(':')[0])])
            return f"{a}:{b}"

        res = pd.DataFrame(valid['text'].apply(lambda x: 1 if (x.count('-') > 0) or (x.count('—') > 0) else 0))
        res.columns = ['class']
        res['text'] = valid['text']
        res['V'] = res['text'].apply(lambda x: str(max(x.find('-'), x.find('—'))) + ':' + str(max(x.find('-'), x.find('—'))))
        res['cV'] = res.apply(lambda_cv, axis=1)
        res['class'] = res.apply(lambda x: 0 if x['class'] == 0 else (1 if x['V'] != '-1:-1' else 0), axis=1)
        return res


class GappingHyphenBaseline(BaseSolution):
    def create_solutions(self):
        sol = GappingHyphenBaselineSolution()
        return [(sol.get_params(), sol)]


if __name__ == '__main__':
    clf = GappingHyphenBaseline()
    clf.score_solutions(mode='gapping')
