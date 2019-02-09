import dataset
from gapping.agrr_metrics import binary_metrics, gapping_metrics


class BaseSolution():
    def __init__(self):
        pass;

    def create_solutions(self): # переопределить в производном классе
        pass;

    def score_solutions(self, mode): # binary, gapping, full
        data = dataset.load_df(('train', 'valid')) # пока что
        solutions = self.create_solutions()
        # fillna?
        res = self.evaluate_dev(solutions=solutions, data=data, mode=mode)
        print(res)

    def evaluate_dev(self, solutions, data, mode):
        if (mode == 'binary'):
            res = []
            for description, solution in solutions:
                tmp_metrics = binary_metrics(data['valid']['class'], solution['class'])
                res.append((description, tmp_metrics))
            return res
        if (mode == 'gapping'):
            res = []
            for description, solution in solutions:
                tmp_metrics = gapping_metrics(data['valid'], solution, resolution=False)
                res.append((description, tmp_metrics))
            return res
        if (mode == 'full'):
            res = []
            for description, solution in solutions:
                tmp_metrics = gapping_metrics(data['valid'], solution, resolution=True)
                res.append((description, tmp_metrics))
            return res
