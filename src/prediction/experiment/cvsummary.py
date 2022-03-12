class CVSummary():
    def __init__(self, metrics: list) -> None:
        self.metrics = metrics
        self.cv_scores = {metric.__name__: [] for metric in self.metrics}
        

    def score_cv(self, y_test, y_hat) -> None:
        for metric in self.metrics:
            self.cv_scores[metric.__name__].append(metric(y_test, y_hat))

    def display_stats(self):
        for idx, score in enumerate(self.cv_scores):
            print(f'\nIteration - {idx+1}')
            for k, v in score.items():
                print(f'Metric: {k} \t\t\t Score: {v}')