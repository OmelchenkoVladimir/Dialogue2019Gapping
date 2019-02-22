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
                solution.fit(data['train'])
                sol_res = solution.predict(data['valid'])
                tmp_metrics = binary_metrics(data['valid']['class'], sol_res['class'])
                res.append((description, tmp_metrics))
            return res
        if (mode == 'gapping'):
            res = []
            for description, solution in solutions:
                solution.fit(data['train'])
                sol_res = solution.predict(data['valid'])
                sol_res.to_csv('debug/res.csv', sep='\t', index=None)
                tmp_metrics = gapping_metrics(data['valid'], sol_res, resolution=True)
                res.append((description, tmp_metrics))
            return res
        if (mode == 'full'):
            res = []
            for description, solution in solutions:
                solution.fit(data['train'])
                sol_res = solution.predict(data['valid'])
                tmp_metrics = gapping_metrics(data['valid'], sol_res, resolution=False)
                res.append((description, tmp_metrics))
            return res

    def predict_test(self):
        solutions = self.create_solutions()
        data = dataset.load_df(('train', 'valid', 'test'))
        assert len(solutions) == 1
        for solution in solutions:
            solution.fit(data['train'])
            res = solution.predict(data['valid']) #  сменить на тест
            res.to_csv('results/res.csv', index=None, sep='\t')
