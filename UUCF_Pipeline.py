from classifiers.UUCF_classifier import UUCF_classifier
from data_preprocessing import DataPreprocessing
from data_preprocessing import split_data
from Result import Result

TRAIN_DATA_PATH = "data/train/one_hour"

def transform( X,columns):

    is_na = None

    for column in columns:
        X.loc[~X[column].isna(), column] = 1
        X.loc[X[column].isna(), column] = 0

        if is_na is not None:
            is_na = is_na & X[column] == 0
        else:
            is_na = X[column] == 0

    return X


def main():
    data_preprocessing = DataPreprocessing(TRAIN_DATA_PATH)
    X = data_preprocessing.read_train_data()
    X=transform(X,['like_timestamp'])
    X_train, X_test, y_train, y_test = split_data(X, X['like_timestamp'], test_size=0.2)
    print(len(X_test))
    UUCF = UUCF_classifier()
    UUCF.train(X_train, y_train, 'like_timestamp')
    predictions = UUCF.predict_proba(X_test)
    result = Result('UUCF', UUCF)
    result.calculate_and_store_metrics(y_test, predictions)
    result.store_result()


if __name__ == "__main__":
    main()
