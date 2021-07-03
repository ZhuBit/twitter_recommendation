from sklearn.ensemble import AdaBoostClassifier

import data_preprocessing as dp
import utils
from Result import Result
from torch.utils.data import Dataset, DataLoader
import numpy as np
from batches import TimeSeriesDataSet
from classifiers.neural_network import NeuralNetworkClassifier, NeuralNetworkNet
from classifiers.xgboost_classifier import XGBoostClassifier

train_data_path = "data/one_hour"


def load_classifiers(num_of_inputs):
    xgb_classifier = XGBoostClassifier()
    net = NeuralNetworkNet(num_of_inputs)
    neural_network_classifier = NeuralNetworkClassifier(net)
    return [
        #{'model': AdaBoostClassifier(n_estimators=10, random_state=0), 'name': 'AdaBoost 1'},
        #{'model': AdaBoostClassifier(n_estimators=20,  random_state=1), 'name': 'AdaBoost 2'},
        #{'model': xgb_classifier.classifier, 'name': xgb_classifier.name},
        {'model':neural_network_classifier, 'name':neural_network_classifier.name}
    ]




if __name__=="__main__":
    data = dp.read_train_data(train_data_path)
    features, targets, ids = dp.preprocess_data(data)
    X_train, X_test, y_train, y_test = dp.split_data(features, targets['reply_timestamp'], test_size=0.2)

    classifiers = load_classifiers(X_train.shape[1])

    for classifier in classifiers:
        model = classifier['model']
        result = Result(classifier['name'], model, str(model.classifier.parameters()))
        print('start: {}'.format(classifier['name']))

        # The Dataloader class handles all the shuffles for you
        loader = iter(DataLoader(TimeSeriesDataSet(X_train, y_train), batch_size=32, shuffle=True))
        try:
            while True:
                x, y = loader.next()
                if x is None or y is None:
                    break
                x=x.cpu().detach().numpy()
                y=y.cpu().detach().numpy()
                y = np.reshape(y, (len(y),))
                model.train(x, y)
        except StopIteration:
            pass


        loader = iter(DataLoader(TimeSeriesDataSet(X_test, y_test), batch_size=len(X_test), shuffle=False))
        x_test, y_test = loader.next()

        x_test = x_test.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()
        y_test = np.reshape(y_test, (len(y_test),))

        y_pred = model.predict(x_test)
        y_pred=y_pred.detach().numpy()
        y_pred=np.array(y_pred).flatten()

        result.calculate_and_store_metrics(y_test, y_pred)
        result.store_result()
        utils.store_model(model,  classifier['name'])
