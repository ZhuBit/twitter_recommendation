from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
import utils
from Result import Result
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from data_preprocessing import DataPreprocessing
from data_preprocessing import split_data

TRAIN_DATA_PATH = "data/one_hour.tsv"


def load_classifiers():
    return [
        {'model': RandomForestClassifier(n_estimators=10, min_samples_leaf=1, class_weight={0:1, 1: 100}), 'name': 'Random Forest 1'},
        {'model': RandomForestClassifier(n_estimators=10, min_samples_leaf=2,  class_weight={0:1, 1: 100}), 'name': 'Random Forest 2'},
        {'model': RandomForestClassifier(n_estimators=100, min_samples_leaf=2,  class_weight={0:1, 1: 100}), 'name': 'Random Forest 3'},
        {'model': RandomForestClassifier(n_estimators=1000, min_samples_leaf=2,  class_weight={0:1, 1: 100}), 'name': 'Random Forest 5'},
        {'model': RandomForestClassifier(n_estimators=100, min_samples_leaf=1,  class_weight={0:1, 1: 100}), 'name': 'Random Forest 4'},
        {'model': GradientBoostingClassifier(n_estimators=100, min_samples_leaf=1,  class_weight={0:1, 1: 100}), 'name': 'GBC 1'},
        {'model': GradientBoostingClassifier(n_estimators=100, min_samples_leaf=2,  class_weight={0:1, 1: 100}), 'name': 'GBC 2'},
        # same as in boosting-mlp.py
        {'model': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 5), random_state=1),'name': 'MLP 1'},
        {'model': MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(150, 6), random_state=1,  class_weight={0:1, 1: 100}),'name': 'MLP 2'},
        {'model': MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(200, 7), random_state=1,  class_weight={0:1, 1: 100}),'name': 'MLP 3'},
        {'model': DecisionTreeClassifier(max_depth=50, random_state=1,  class_weight={0:1, 1: 100}), 'name': 'DecisionTree 1'},
        {'model': DecisionTreeClassifier(max_depth=80, random_state=1,  class_weight={0:1, 1: 100}), 'name': 'DecisionTree 2'},
        {'model': DecisionTreeClassifier(max_depth=100, random_state=1,  class_weight={0:1, 1: 100}), 'name': 'DecisionTree 3'},
        {'model': SVC(class_weight={0:1, 1: 1000}), 'name': 'SVM 1'},
        {'model': SVC(kernel='poly', class_weight={0:1, 1: 1000}), 'name': 'SVM 2'},
        {'model': SVC(C=0.1, class_weight={0:1, 1: 1000}), 'name': 'SVM 3'},
        {'model': SVC(kernel='poly',C=0.1,class_weight={0:1, 1: 1000} ), 'name': 'SVM 4'},
        {'model': BaggingClassifier(base_estimator=Perceptron(), n_estimators=1000, bootstrap=True), 'name': 'bagging 1'},
        {'model': BaggingClassifier(base_estimator=Perceptron(), n_estimators=100, bootstrap=True), 'name': 'bagging 2'},
        {'model': BaggingClassifier(base_estimator=Perceptron(), n_estimators=100, bootstrap=True, max_features=5), 'name': 'bagging 3'},
        {'model': BaggingClassifier(base_estimator=Perceptron(), n_estimators=1000, bootstrap=True, max_features=5), 'name': 'bagging 4'},
        {'model': BaggingClassifier(base_estimator=Perceptron(), n_estimators=1000, bootstrap=False), 'name': 'bagging 5'},

    ]


if __name__=="__main__":
    ########################################
    # READ_DATA
    ########################################

    data_preprocessing = DataPreprocessing(TRAIN_DATA_PATH)
    features, targets = data_preprocessing.get_processed_data()

    X_train, X_test, y_train, y_test = split_data(features, targets['like_timestamp'], test_size=0.2)

    for classifier in load_classifiers():
        model = classifier['model']

        result = Result('with numeric values, class weights 100, like', model, str(model.get_params()))

        print('start: {}'.format(classifier['name']))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        result.calculate_and_store_metrics(y_test, y_pred)
        result.store_result()
        utils.store_model(model,  classifier['name'])

