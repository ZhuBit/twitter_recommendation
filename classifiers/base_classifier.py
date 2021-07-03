class BaseClassifier:
    def __init__(self, name: str):
        super(BaseClassifier, self).__init__()
        self.name = name
        self.classifier = None
        self.search_parameters = None

    def get_classifier(self):
        return self.classifier

    def get_search_parameters(self):
        return self.search_parameters

    def set_search_parameters(self, search_parameters):
        self.search_parameters = search_parameters

    def predict(self, X_test):
        pass

    def get_name(self):
        return self.name

    def train(self, X, y):
        raise RuntimeError('Not implemented')