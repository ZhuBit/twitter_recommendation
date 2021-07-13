from classifiers.base_classifier import BaseClassifier

class UUCF_classifier(BaseClassifier):
    def __init__(self):
        super().__init__('UUCF')
        self.classifer=None

    def train(self,X,y):

