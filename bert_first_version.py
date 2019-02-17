import pandas as pd
import pymorphy2
from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer, BasicTokenizer

from base_solution import BaseSolution
from sklearn.base import BaseEstimator
from layers import pandas_top_layer


class BertFirstSolution(BaseEstimator):
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
        res.to_csv('bert/bert_first_5.csv')
        return res


class BertFirst(BaseSolution):
    def create_solutions(self):  # TODO: чтение параметров из config'а
        model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
        model.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        sentence_tokenizer = BasicTokenizer()
        threshold = 5 # default, будет гиперпараметром из config'а
        sol = BertFirstSolution(tokenizer, model, sentence_tokenizer, threshold)
        return [(sol.get_params(), sol)]


if __name__ == '__main__':
    clf = BertFirst()
    clf.score_solutions(mode='gapping')
