team_name = "team_6"  # your team name e.g. 'team_1'
team_members = [("Antonio Filipovic", "12005824"),
                ("Dzan Operta", "11935976"),
                ("Filip Sekerija", "12005824"),
                ("Martin Suchon", "12040562"),
                ("Kaul Saroj", "12003943",
                 ("Omar Drljevic", ""))]

print(team_name)
print(team_members)

path_to_data = '~/shared/data/project/training/'
path_to_data = 'data/train/'  ##TODO delete
dataset_type = 'one_hour'  # all_sorted, one_day, one_hour, one_week

import os
import re
import csv
import datetime

from model import reply_pred_model_RF, retweet_pred_model_RF, quote_pred_model_RF, fav_pred_model_RF,\
    reply_pred_model_NN, retweet_pred_model_NN, quote_pred_model_NN, fav_pred_model_NN, \
    reply_pred_model_MLP, retweet_pred_model_MLP, quote_pred_model_MLP, fav_pred_model_MLP,\
    fav_pred_model_UU

all_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",
                "engaged_with_user_following_count", "engaged_with_user_is_verified",
                "engaged_with_user_account_creation", "engaging_user_id", "enaging_user_follower_count",
                "enaging_user_following_count",
                "enaging_user_is_verified", "enaging_user_account_creation", "engagee_follows_engager"]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))


def parse_input_line(line):
    features = line  # .split("\x01")
    tweet_id = features[all_features_to_idx['tweet_id']]
    user_id = features[all_features_to_idx['engaging_user_id']]
    # use all features for data preprocessing
    input_feats = features
    tweet_timestamp = features[all_features_to_idx['tweet_timestamp']]
    return tweet_id, user_id, input_feats, tweet_timestamp


def evaluate_test_set(key):
    expanded_path = os.path.expanduser(path_to_data)
    part_files = [os.path.join(expanded_path, f) for f in os.listdir(expanded_path) if dataset_type in f]
    part_files = sorted(part_files, key=lambda x: x[-5:])
    with open('results.csv', 'w') as output:
        for file in part_files:
            with open(file, 'r') as f:
                linereader = csv.reader(f, delimiter='\x01')
                last_timestamp = None
                for row in linereader:
                    tweet_id, user_id, features, tweet_timestamp = parse_input_line(row)
                    if key == 'RF':
                        reply_pred = reply_pred_model_RF(features)  # reply_model
                        retweet_pred = retweet_pred_model_RF(features)  # retweet_model
                        quote_pred = quote_pred_model_RF(features)  # pred_model
                        fav_pred = fav_pred_model_RF(features)  # fav_model
                    elif key == 'NN':
                        reply_pred = reply_pred_model_NN(features)  # reply_model
                        retweet_pred = retweet_pred_model_NN(features)  # retweet_model
                        quote_pred = quote_pred_model_NN(features)  # pred_model
                        fav_pred = fav_pred_model_NN(features)  # fav_model
                    elif key == 'MLP':
                        reply_pred = reply_pred_model_MLP(features)  # reply_model
                        retweet_pred = retweet_pred_model_MLP(features)  # retweet_model
                        quote_pred = quote_pred_model_MLP(features)  # pred_model
                        fav_pred = fav_pred_model_MLP(features)  # fav_model
                    elif key == 'UU':
                        """reply_pred = reply_pred_model(features)  # reply_model
                        retweet_pred = retweet_pred_model(features)  # retweet_model
                        quote_pred = quote_pred_model(features)  # pred_model"""
                        fav_pred = fav_pred_model_UU(features)  # fav_model
                    print(str(tweet_timestamp))
                    print(str(reply_pred) + " " + str(retweet_pred) + " " + str(quote_pred) + " " + str(fav_pred))
                    output.write(f'{tweet_id},{user_id},{reply_pred},{retweet_pred},{quote_pred},{fav_pred}\n')


# Choose which model
# The best performing is Random Forest
keys = [ 'RF', 'NN', 'MLP', 'UU']
key = 'RF'
evaluate_test_set(key)