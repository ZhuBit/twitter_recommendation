import numpy as np
import pandas as pd
# import neural_network_pipeline as nnp
import data_preprocessing as dp
from joblib import load

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


##########################


def reply_pred_model(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/reply/Random Forest 3')
    result = loaded_model.predict(X_test)
    return result

    # TODO fill in your implementation of the model


    ######
    #NEURAL NETWORK PART
    #y_pred = neural_network_pipeline.perform_prediction(X_test)
    #return y_pred

    # return np.random.rand()

def retweet_pred_model(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/reply/Random Forest 3')
    result = loaded_model.predict(X_test)
    return result

def quote_pred_model(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/retweet_with_comment/Random Forest 3')
    result = loaded_model.predict(X_test)
    return result

def fav_pred_model(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/like/Random Forest 3')
    result = loaded_model.predict(X_test)
    return result