from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import data_preprocessing as dp
import utils
from Result import Result
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


train_data_path = "data/one_hour"


def load_classifiers():
    return [
        {'model': RandomForestClassifier(n_estimators=10, min_samples_leaf=1), 'name': 'Random Forest 1'},
        {'model': RandomForestClassifier(n_estimators=10, min_samples_leaf=2), 'name': 'Random Forest 2'},
        {'model': RandomForestClassifier(n_estimators=100, min_samples_leaf=2), 'name': 'Random Forest 3'},
        {'model': RandomForestClassifier(n_estimators=100, min_samples_leaf=1), 'name': 'Random Forest 4'},
        {'model': GradientBoostingClassifier(n_estimators=100, min_samples_leaf=1), 'name': 'GBC 1'},
        {'model': GradientBoostingClassifier(n_estimators=100, min_samples_leaf=2), 'name': 'GBC 2'},
        # same as in boosting.py
        {'model': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 5), random_state=1),'name': 'MLP 1'},
        {'model': MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(150, 6), random_state=1),'name': 'MLP 2'},
        {'model': MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(200, 7), random_state=1),'name': 'MLP 3'},
        {'model': DecisionTreeClassifier(max_depth=50, random_state=1), 'name': 'DecisionTree 1'},
        {'model': DecisionTreeClassifier(max_depth=80, random_state=1), 'name': 'DecisionTree 2'},
        {'model': DecisionTreeClassifier(max_depth=100, random_state=1), 'name': 'DecisionTree 3'},
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

