import numpy as np
import pandas as pd
# import neural_network_pipeline as nnp
import data_preprocessing as dp
from joblib import load

'''
You need to structure your implementation in a way that in this class the prediction
of the different engagement types of your model are called. Please replace the 
random return placeholder with you actual implementation.

You can train different models for each engagement type or you train one which is able
to predicte multiple classes.
'''

def reply_pred_model(input_features):
    X_test, y_test = dp.DataPreprocessing('').preprocess_row(input_features)
    loaded_model = load('trained_models/reply/Random Forest 3')
    result = loaded_model.predict(X_test)
    return result
    # TODO fill in your implementation of the model
    #print(input_features)
    # neural_network_pipeline = nnp.NeuralNetworkPipeline('reply_timestamp')
    # y_pred = neural_network_pipeline.train_neural_network()
    # input_features_split = input_features.split('\t')
    # print(len(input_features_split))
    # dp.preprocess_data(pd.DataFrame(input_features))
    # print("input features", input_features)
    #
    # return neural_network_pipeline.perform_prediction(input_features)
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