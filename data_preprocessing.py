import pandas as pd
pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 1000)
from sklearn.pipeline import Pipeline
import feature_engineering as fe
from sklearn.model_selection import train_test_split

train_data_path = "data/one_hour.tsv"


def read_train_data(train_data_path):
    column_names = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                    "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id",
                    "engaged_with_user_follower_count", "engaged_with_user_following_count",
                    "engaged_with_user_is_verified",
                    "engaged_with_user_account_creation", "engaging_user_id", "engaging_user_follower_count",
                    "engaging_user_following_count",
                    "engaging_user_is_verified", "engaging_user_account_creation", "engaged_follows_engaging",
                    "reply_timestamp",
                    "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]

    return pd.read_csv(train_data_path, header=None, names=column_names, delimiter='\x01')


def preprocess_data(data):
    df = data.copy()

    preprocessing_pipeline = Pipeline(
        [
            ('target_encoder', fe.TargetEncoder(columns=["reply_timestamp", "retweet_timestamp",
                                                             "retweet_with_comment_timestamp", "like_timestamp"])),

            # splitting list features and counting the number of elements in the lists
            ('tab_list_spliter', fe.ListTabSplitter(columns=["hashtags", "present_media", "present_links",
                                                             "present_domains"])),
            ('count_elements', fe.ListCountEncoder(columns=["hashtags", "present_links", "present_domains",
                                                            "present_media"])),

            # one hot encode categorical columns
            ('one_hot_encoder', fe.OneHotEncoder(columns=["tweet_type", "language", "engaged_with_user_is_verified",
                                                          "engaging_user_is_verified", "engaged_follows_engaging"])),

            # TODO 1. check what gives better results, quantiles or pure count values
            # create categorical bins based on numeric quantiles
            # ('numeric_bins', fe.NumericQuantileBucketOneHotEncoder(columns=["engaged_with_user_follower_count",
            #                                                                 "engaged_with_user_following_count",
            #                                                                 "engaging_user_follower_count",
            #                                                                 "engaging_user_following_count"])),
        ]
    )

    df = preprocessing_pipeline.fit_transform(df)

    labels = df[["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]]
    ids = df[["tweet_id", "engaging_user_id", "engaged_with_user_id"]]

    df.drop(
        columns=[
            "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp",

            "tweet_id", "engaging_user_id", "engaged_with_user_id",

            # not using text tokens
            "text_tokens",

            # not using timestamps
            "tweet_timestamp",

            # drop preprocessed columns
            "hashtags", "present_links", "present_domains", "present_media", "tweet_type", "language",
            "engaged_with_user_is_verified", "engaging_user_is_verified", "engaged_follows_engaging"

            # TODO 1.
            # "engaged_with_user_follower_count", "engaged_with_user_following_count",
            # "engaging_user_follower_count", "engaging_user_following_count"
        ],
        inplace=True)

    return df, labels, ids


def split_data(train_data, train_labels, test_size):
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=test_size)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    df = read_train_data(train_data_path)
    df, labels, ids = preprocess_data(df)
    print(df.columns)
