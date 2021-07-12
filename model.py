import numpy as np
import pandas as pd

import neural_network_pipeline as nnp
import data_preprocessing as dp
'''
You need to structure your implementation in a way that in this class the prediction
of the different engagement types of your model are called. Please replace the 
random return placeholder with you actual implementation.

You can train different models for each engagement type or you train one which is able
to predicte multiple classes.
'''

def reply_pred_model(input_features):
    # TODO fill in your implementation of the model
    #print(input_features)
    neural_network_pipeline = nnp.NeuralNetworkPipeline('reply_timestamp')
    y_pred = neural_network_pipeline.train_neural_network()
    input_features_split = input_features.split('\t')
    print(len(input_features_split))
    dp.preprocess_data(pd.DataFrame(input_features))
    print("input features", input_features)

    return neural_network_pipeline.perform_prediction(input_features)

def retweet_pred_model(input_features):
    # TODO fill in your implementation of the model
    return np.random.rand()

def quote_pred_model(input_features):
    # TODO fill in your implementation of the model
    return np.random.rand()

def fav_pred_model(input_features):
    # TODO fill in your implementation of the model
    return np.random.rand()