from sklearn.ensemble import AdaBoostClassifier

import data_preprocessing as dp
import utils
from Result import Result
from torch.utils.data import Dataset, DataLoader
import numpy as np
from batches import TimeSeriesDataSet
from classifiers.xgboost_classifier import XGBoostClassifier

train_data_path = "data/one_hour"


def load_classifiers(num_of_inputs):
    xgb_classifier = XGBoostClassifier()
    return [
        #{'model': AdaBoostClassifier(n_estimators=10, random_state=0), 'name': 'AdaBoost 1'},
        #{'model': AdaBoostClassifier(n_estimators=20,  random_state=1), 'name': 'AdaBoost 2'},
        {'model': xgb_classifier.classifier, 'name': xgb_classifier.name},
    ]




if __name__=="__main__":
    data = dp.read_train_data(train_data_path)
    features, targets, ids = dp.preprocess_data(data)
    X_train, X_test, y_train, y_test = dp.split_data(features, targets['reply_timestamp'], test_size=0.2)

    print(np.array(X_train.iloc[0]))
    print(np.array(y_train.iloc[0]))
    classifiers = load_classifiers(X_train.shape[1])

    for classifier in classifiers:
        model = classifier['model']
        result = Result(classifier['name'], model, str(model.get_params()))
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
                #print(x)
                #print("y",y.shape)
                model.fit(x, y)
        except StopIteration:
            pass

        y_pred = model.predict(X_test)

        result.calculate_and_store_metrics(y_test, y_pred)
        result.store_result()
        utils.store_model(model,  classifier['name'])