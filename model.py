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

TARGET = "reply_timestamp"

# LOAD ONCE ALL MODELS USED
##########################

# NEURAL NETWORK PART

neural_network_pipeline = NeuralNetworkPipeline(TARGET)
neural_network_pipeline.load_model('best_model_{}.pt'.format(TARGET), neural_network_pipeline.X_train.shape[1])

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
    loaded_model = load('trained_models/reply/Random Forest 3')
    result = loaded_model.predict(X_test)
    return result


def quote_pred_model_RF(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/reply/Random Forest 3')
    result = loaded_model.predict(X_test)
    return result


def fav_pred_model_RF(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/reply/Random Forest 3')
    result = loaded_model.predict(X_test)
    return result

# Preditions for Neural-Network

def reply_pred_model_NN(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    y_pred = neural_network_pipeline.perform_prediction(X_test)
    return y_pred


def retweet_pred_model_NN(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    y_pred = neural_network_pipeline.perform_prediction(X_test)
    return None


def quote_pred_model_NN(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    y_pred = neural_network_pipeline.perform_prediction(X_test)
    return None


def fav_pred_model_NN(input_features):
    return None

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