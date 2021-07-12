from sklearn.ensemble import AdaBoostClassifier

import data_preprocessing as dp
import utils
from Result import Result
from torch.utils.data import Dataset, DataLoader
import numpy as np
from batches import TimeSeriesDataSet
from classifiers.xgboost_classifier import XGBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

train_data_path = "data/train/one_hour"

BATCH_SIZE = 128


def load_classifiers(num_of_inputs):
    xgb_classifier = XGBoostClassifier()
    return [
        {'model': xgb_classifier.classifier, 'name': xgb_classifier.name},
        #{'model': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 5), random_state=1),
        # 'name': 'MLP 1'},
        {'model': MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(150, 6), random_state=1),
         'name': 'MLP 2'},
        {'model': MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(200, 7), random_state=1),
         'name': 'MLP 3'},
        {'model': DecisionTreeClassifier(max_depth=50, random_state=1), 'name': 'DecisionTree 1'},
        {'model': DecisionTreeClassifier(max_depth=80, random_state=1), 'name': 'DecisionTree 2'},
        {'model': DecisionTreeClassifier(max_depth=100, random_state=1), 'name': 'DecisionTree 3'},
    ]


if __name__ == "__main__":
    data = dp.read_train_data(train_data_path)
    features, targets, ids = dp.preprocess_data(data)
    X_train, X_test, y_train, y_test = dp.split_data(features, targets['reply_timestamp'], test_size=0.2)

    # print(np.array(X_train.iloc[0]))
    # print(np.array(y_train.iloc[0]))
    classifiers = load_classifiers(X_train.shape[1])

    data_length = X_train.shape[0]
    for classifier in classifiers:
        model = classifier['model']
        result = Result(classifier['name'], model, str(model.get_params()))
        print('start: {}'.format(classifier['name']))

        data_trained = 0
        # The Dataloader class handles all the shuffles for you
        loader = iter(DataLoader(TimeSeriesDataSet(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False))
        try:
            while True:
                x, y = loader.next()
                if x is None or y is None:
                    break
                x = x.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                y = np.reshape(y, (len(y),))
                # print(x)
                # print("y",y.shape)
                model.fit(x, y)
                data_trained += BATCH_SIZE
                print('classifier: {0}, status:{1:8.2f}%'.format(classifier['name'], data_trained / data_length * 100 ))
        except StopIteration:
            pass

        y_pred = model.predict(X_test)

        result.calculate_and_store_metrics(y_test, y_pred)
        result.store_result()
        utils.store_model(model, classifier['name'])
