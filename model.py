import numpy as np
import pandas as pd
# import neural_network_pipeline as nnp
import data_preprocessing as dp
from joblib import load
from classifiers.UUCF_classifier import UUCF_classifier

from neural_network_pipeline import NeuralNetworkPipeline

'''
You need to structure your implementation in a way that in this class the prediction
of the different engagement types of your model are called. Please replace the 
random return placeholder with you actual implementation.

You can train different models for each engagement type or you train one which is able
to predicte multiple classes.
'''

# LOAD ONCE ALL MODELS USED
##########################

# NEURAL NETWORK PART

#####################
NEURAL_NETWORK_DIR = "neural_network_models"

neural_network_pipeline_reply = NeuralNetworkPipeline("reply_timestamp")
neural_network_pipeline_reply.load_model('{0}/best_model_{1}.pt'.format(NEURAL_NETWORK_DIR,"reply_timestamp"), neural_network_pipeline_reply.X_train.shape[1])

neural_network_pipeline_retweet = NeuralNetworkPipeline("retweet_timestamp")
neural_network_pipeline_retweet.load_model('{0}/best_model_{1}.pt'.format(NEURAL_NETWORK_DIR,"retweet_timestamp"), neural_network_pipeline_retweet.X_train.shape[1])


neural_network_pipeline_retweet_comment = NeuralNetworkPipeline("retweet_with_comment_timestamp")
neural_network_pipeline_retweet_comment.load_model('{0}/best_model_{1}.pt'.format(NEURAL_NETWORK_DIR,"retweet_with_comment_timestamp"), neural_network_pipeline_retweet_comment.X_train.shape[1])


neural_network_pipeline_like = NeuralNetworkPipeline("like_timestamp")
neural_network_pipeline_like.load_model('{0}/best_model_{1}.pt'.format(NEURAL_NETWORK_DIR,"like_timestamp"), neural_network_pipeline_like.X_train.shape[1])
##############################

#UUCF

##########################
# Predictions for Random-Forest
def reply_pred_model_RF(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/reply/Random Forest 3')
    result = loaded_model.predict(X_test)
    return result


def retweet_pred_model_RF(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/retweet/Random Forest 3')
    result = loaded_model.predict(X_test)
    return result


def quote_pred_model_RF(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/retweet_with_comment/Random Forest 3')
    result = loaded_model.predict(X_test)
    return result


def fav_pred_model_RF(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/like/Random Forest 3')
    result = loaded_model.predict(X_test)
    return result


# PREDICTIONS for Neural-Network
def reply_pred_model_NN(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    y_pred = neural_network_pipeline_reply.perform_prediction(X_test)
    return y_pred


def retweet_pred_model_NN(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    y_pred = neural_network_pipeline_retweet.perform_prediction(X_test)
    return y_pred


def quote_pred_model_NN(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    y_pred = neural_network_pipeline_retweet_comment.perform_prediction(X_test)
    return y_pred


def fav_pred_model_NN(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    y_pred = neural_network_pipeline_like.perform_prediction(X_test)
    return y_pred

# Predictions for MLP
def reply_pred_model_MLP(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/MLP')
    result = loaded_model.predict(X_test)
    return result


def retweet_pred_model_MLP(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/MLP')
    result = loaded_model.predict(X_test)
    return result


def quote_pred_model_MLP(input_features):
    return None


def fav_pred_model_MLP(input_features):
    return None


#Prediction for UU
def fav_pred_model_UU(input_features):
    TYPE_OF_ENGAGEMENT='like_timestamp'
    data_preprocessing = dp.DataPreprocessing('~/shared/data/project/training/one_hour')
    X = data_preprocessing.read_train_data()
    X=dp.transform_data_for_uucf(X,TYPE_OF_ENGAGEMENT)
    Y=dp.transform_row_for_uucf(input_features)
    Y=dp.transform_data_for_uucf(Y,TYPE_OF_ENGAGEMENT)
    UUCF = UUCF_classifier()
    UUCF.train(X, X[TYPE_OF_ENGAGEMENT], TYPE_OF_ENGAGEMENT)
    prediction= UUCF.predict_proba(Y)
    return prediction