import pandas as pd

from sklearn.pipeline import Pipeline
import feature_engineering as fe
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 1000)
TRAIN_DATA_PATH = "data/train/one_hour"

COLUMN_NAMES = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id",
                "engaged_with_user_follower_count", "engaged_with_user_following_count",
                "engaged_with_user_is_verified",
                "engaged_with_user_account_creation", "engaging_user_id", "engaging_user_follower_count",
                "engaging_user_following_count",
                "engaging_user_is_verified", "engaging_user_account_creation", "engaged_follows_engaging",
                "reply_timestamp",
                "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]

TARGET_COLUMN_NAMES = ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]
TAB_SPLIT_COLUMNS = ["hashtags", "present_media", "present_links", "present_domains"]
ONE_HOT_ENCODING_COLUMNS = ["tweet_type", "language", "engaged_with_user_is_verified", "engaging_user_is_verified",
                            "engaged_follows_engaging"]


def split_data(train_data, train_labels, test_size) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=test_size)
    return X_train, X_test, y_train, y_test


class DataPreprocessing():
    def __init__(self, train_data_path):
        self.train_data_path = train_data_path
        self.df = None
        self.labels = None

    def read_train_data(self) -> pd.DataFrame:
        return pd.read_csv(self.train_data_path, header=None, names=COLUMN_NAMES, delimiter='\x01')

    def preprocess_data(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        df: pd.DataFrame = data.copy()

        preprocessing_pipeline = Pipeline([
            ('target_encoder', fe.TargetEncoder(columns=TARGET_COLUMN_NAMES)),

            # splitting list features and counting the number of elements in the lists
            ('tab_list_spliter', fe.ListTabSplitter(columns=TAB_SPLIT_COLUMNS)),
            ('count_elements', fe.ListCountEncoder(columns=TAB_SPLIT_COLUMNS)),

            # one hot encode categorical columns
            ('one_hot_encoder', fe.OneHotEncoder(columns=ONE_HOT_ENCODING_COLUMNS)),

            # TODO check what gives better results, quantiles or pure count values
            # create categorical bins based on numeric quantiles
            # ('numeric_bins', fe.NumericQuantileBucketOneHotEncoder(columns=["engaged_with_user_follower_count",
            #                                                                 "engaged_with_user_following_count",
            #                                                                 "engaging_user_follower_count",
            #                                                                 "engaging_user_following_count"])),
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

        df = df.drop(columns=columns_to_drop)

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
