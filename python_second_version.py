import pandas as pd
import pymorphy2
from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer, BasicTokenizer

from base_solution import BaseSolution
from sklearn.base import BaseEstimator
from layers import pandas_top_layer


class BertSecondSolution(BaseEstimator):
    def __init__(self, tokenizer, model, sentence_tokenizer, threshold):
        self.tokenizer = tokenizer
        self.model = model
        self.sentence_tokenizer = sentence_tokenizer
        self.threshold = threshold
        self.morph = pymorphy2.MorphAnalyzer()

    def fit(self, train):
        pass;

    def predict(self, valid):
        valid_applied = valid.apply(pandas_top_layer, args=(self.tokenizer, self.model, self.sentence_tokenizer, self.morph, self.threshold), axis = 1)
        res = valid_applied[['text', 'res_class', 'res_cV', 'res_V']].rename({'res_class':'class', 'res_cV':'cV', 'res_V':'V'}, axis = 1)
        res.to_csv('bert/-----.csv')
        return res


class BertSecond(BaseSolution):
    def create_solutions(self):  # TODO: чтение параметров из config'а
        model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
        model.eval()
        model.to('cuda')
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        sentence_tokenizer = BasicTokenizer(do_lower_case=False)
        threshold = 25 # default, будет гиперпараметром из config'а
        sol = BertSecondSolution(tokenizer, model, sentence_tokenizer, threshold)
        return [(sol.get_params(), sol)]


if __name__ == '__main__':
    clf = BertSecond()
    clf.score_solutions(mode='gapping')
