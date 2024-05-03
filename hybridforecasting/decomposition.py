class Decomposition:
    def __init__(self, data, model, **decomposition_kwargs):
        self.data = data
        self.model = model(**decomposition_kwargs)

    def fit(self):
        return self.model.fit()
