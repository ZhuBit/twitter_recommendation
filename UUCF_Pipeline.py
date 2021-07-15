from classifiers.UUCF_classifier import UUCF_classifier
from data_preprocessing import DataPreprocessing
from data_preprocessing import split_data
from Result import Result

TRAIN_DATA_PATH = "data/train/one_hour"
VALIDATION_DATA_PATH="data/validation/one_hour"
TYPE_OF_ENGAGEMENT='retweet_timestamp'


def main():
    data_preprocessing = DataPreprocessing(TRAIN_DATA_PATH)
    X = data_preprocessing.read_train_data()
    X=transform(X,TYPE_OF_ENGAGEMENT)
    validation_data = DataPreprocessing(VALIDATION_DATA_PATH)
    Y=validation_data.read_train_data()
    Y=transform(Y,TYPE_OF_ENGAGEMENT)
    #X_train, X_test, y_train, y_test = split_data(X, X['like_timestamp'], test_size=0.2)
    print(len(Y))
    UUCF = UUCF_classifier()
    UUCF.train(X, X[TYPE_OF_ENGAGEMENT], TYPE_OF_ENGAGEMENT)
    predictions = UUCF.predict_proba(Y)
    result = Result('UUCF', TYPE_OF_ENGAGEMENT)
    result.calculate_and_store_metrics(Y[TYPE_OF_ENGAGEMENT], predictions)
    result.store_result()


if __name__ == "__main__":
    main()
