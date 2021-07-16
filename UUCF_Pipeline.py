from classifiers.UUCF_classifier import UUCF_classifier
from data_preprocessing import DataPreprocessing
import data_preprocessing
from data_preprocessing import split_data
from Result import Result

TRAIN_DATA_PATH = "data/train/one_hour"
VALIDATION_DATA_PATH="data/validation/one_hour"
TYPE_OF_ENGAGEMENT='retweet_timestamp'


def main():
    DP = DataPreprocessing(TRAIN_DATA_PATH)
    X = DP.read_train_data()
    X=data_preprocessing.transform_data_for_uucf(X,TYPE_OF_ENGAGEMENT)
    validation_data = DataPreprocessing(VALIDATION_DATA_PATH)
    Y=validation_data.read_train_data()
    Y=data_preprocessing.transform_data_for_uucf(Y,TYPE_OF_ENGAGEMENT)
    print(len(Y))
    UUCF = UUCF_classifier()
    UUCF.train(X, X[TYPE_OF_ENGAGEMENT], TYPE_OF_ENGAGEMENT)
    predictions = UUCF.predict_proba(Y)
    result = Result('UUCF', TYPE_OF_ENGAGEMENT)
    result.calculate_and_store_metrics(Y[TYPE_OF_ENGAGEMENT], predictions)
    result.store_result()


if __name__ == "__main__":
    main()