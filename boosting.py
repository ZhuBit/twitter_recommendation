from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import data_preprocessing as dp
import utils
from Result import Result


train_data_path = "data/one_hour.tsv"


def load_classifiers():
    return [
        {'model': AdaBoostClassifier(n_estimators=10, random_state=0), 'name': 'AdaBoost 1'},
        {'model': AdaBoostClassifier(n_estimators=20,  random_state=1), 'name': 'AdaBoost 2'},
    ]


if __name__=="__main__":
    data = dp.read_train_data(train_data_path)
    features, targets, ids = dp.preprocess_data(data)
    X_train, X_test, y_train, y_test = dp.split_data(features, targets['reply_timestamp'], test_size=0.2)

    for classifier in load_classifiers():
        model = classifier['model']

        result = Result(classifier['name'], model, str(model.get_params()))

        print('start: {}'.format(classifier['name']))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        result.calculate_and_store_metrics(y_test, y_pred)
        result.store_result()
        utils.store_model(model,  classifier['name'])
