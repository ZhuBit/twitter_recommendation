import pandas as pd

from sklearn.pipeline import Pipeline

from feature_transformer import *

from sklearn.model_selection import train_test_split

import numpy as np

pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 1000)
TRAIN_DATA_PATH = "data/one_hour.tsv"

COLUMN_NAMES = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id",
                "engaged_with_user_follower_count", "engaged_with_user_following_count",
                "engaged_with_user_is_verified", "engaged_with_user_account_creation", "engaging_user_id",
                "engaging_user_follower_count",
                "engaging_user_following_count", "engaging_user_is_verified", "engaging_user_account_creation",
                "engaged_follows_engaging",
                "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]

TARGET_COLUMN_NAMES = ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]
TAB_SPLIT_COLUMNS = ["hashtags", "present_media", "present_links", "present_domains"]
ONE_HOT_ENCODING_COLUMNS = ["tweet_type", "language", "engaged_with_user_is_verified", "engaging_user_is_verified",
                            "engaged_follows_engaging"]

PREPROCESSED_COLUMN_NAMES = ['engaged_with_user_follower_count', 'engaged_with_user_following_count', 'engaged_with_user_account_creation', 'engaging_user_follower_count', 'engaging_user_following_count', 'engaging_user_account_creation', 'hashtags_count', 'present_media_count', 'present_links_count', 'present_domains_count', 'tweet_type_0', 'tweet_type_1', 'tweet_type_2', 'language_0', 'language_1', 'language_2', 'language_3', 'language_4', 'language_5', 'language_6', 'language_7', 'language_8', 'language_9', 'language_10', 'language_11', 'language_12', 'language_13', 'language_14', 'language_15', 'language_16', 'language_17', 'language_18', 'language_19', 'language_20', 'language_21', 'language_22', 'language_23', 'language_24', 'language_25', 'language_26', 'language_27', 'language_28', 'language_29', 'language_30', 'language_31', 'language_32', 'language_33', 'language_34', 'language_35', 'language_36', 'language_37', 'language_38', 'language_39', 'language_40', 'language_41', 'language_42',
       'language_43', 'language_44', 'language_45', 'language_46', 'language_47', 'language_48', 'language_49', 'language_50', 'language_51', 'language_52', 'language_53', 'language_54', 'language_55', 'language_56', 'language_57', 'language_58', 'language_59', 'engaged_with_user_is_verified_0', 'engaged_with_user_is_verified_1', 'engaging_user_is_verified_0', 'engaging_user_is_verified_1', 'engaged_follows_engaging_0', 'engaged_follows_engaging_1']


def split_data(train_data, train_labels, test_size) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=test_size)
    return X_train, X_test, y_train, y_test


def transform_data_for_uucf(X, column):

    is_na = None
    X.loc[~X[column].isna(), column] = 1
    X.loc[X[column].isna(), column] = 0

    if is_na is not None:
        is_na = is_na & X[column] == 0
    else:
        is_na = X[column] == 0

    return X

def transform_row_for_uucf( input_row):
    df = pd.DataFrame(columns=COLUMN_NAMES)
    df.loc[0] = input_row
    return df


class DataPreprocessing():
    def __init__(self, train_data_path):
        self.train_data_path = train_data_path
        self.df = None
        self.labels = None

    def read_train_data(self) -> pd.DataFrame:
        return pd.read_csv(self.train_data_path, header=None, names=COLUMN_NAMES, delimiter='\x01')

    def preprocess_row(self, input_row):
        df = pd.DataFrame(columns=COLUMN_NAMES)
        df.loc[0] = input_row
        return self.preprocess_data(df)

    def preprocess_data(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        df: pd.DataFrame = data.copy()

        target_transformer = Pipeline(steps=[('target_encoder', TargetEncoder(columns=TARGET_COLUMN_NAMES))])

        preprocessing_pipeline = Pipeline([
            ('target_transformer', target_transformer),

            # splitting list features and counting the number of elements in the lists
            ('tab_list_spliter', ListTabSplitter(columns=TAB_SPLIT_COLUMNS)),
            ('count_elements', ListCountEncoder(columns=TAB_SPLIT_COLUMNS)),

            # one hot encode categorical columns
            ('one_hot_encoder', OneHotEncoder(columns=ONE_HOT_ENCODING_COLUMNS)),

            # create categorical bins based on numeric quantiles -> better results on pure numeric fields
            # ('numeric_bins', NumericQuantileBucketOneHotEncoder(columns=["engaged_with_user_follower_count", "engaged_with_user_following_count","engaging_user_follower_count", "engaging_user_following_count"])),
        ]
        )

        df: pd.DataFrame = preprocessing_pipeline.fit_transform(df)
        labels: pd.DataFrame = df[
            ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]]

        # ids: pd.DataFrame = df[["tweet_id", "engaging_user_id", "engaged_with_user_id"]]

        #####################
        # DROP UNUSED
        #####################
        columns_to_drop = TARGET_COLUMN_NAMES.copy()
        columns_to_drop.extend(["tweet_id", "engaging_user_id", "engaged_with_user_id"])

        # not using text tokens
        columns_to_drop.extend(["text_tokens"])

        # not using timestamps
        columns_to_drop.extend(["tweet_timestamp"])

        columns_to_drop.extend(
            ["hashtags", "present_links", "present_domains", "present_media", "tweet_type", "language",
             "engaged_with_user_is_verified", "engaging_user_is_verified", "engaged_follows_engaging"])

        # columns_to_drop.extend(["engaged_with_user_follower_count", "engaged_with_user_following_count",
        # "engaging_user_follower_count", "engaging_user_following_count"])
        df = df.drop(columns=columns_to_drop)
        df = df.reindex(labels=PREPROCESSED_COLUMN_NAMES,axis=1)
        df = df.fillna(0)
        return df, labels

    def get_processed_data(self):
        if self.df is None:
            self.unprocessed_df: pd.DataFrame = self.read_train_data()
            self.df, self.labels = self.preprocess_data(self.unprocessed_df)
        return self.df, self.labels


def main():
    data_preprocessing = DataPreprocessing(TRAIN_DATA_PATH)
    df, labels = data_preprocessing.get_processed_data()
    print(labels.iloc[0])


if __name__ == '__main__':
    main()
