from sklearn.preprocessing import StandardScaler
import data_preprocessing as dp
import utils
from Result import Result
from torch.utils.data import Dataset, DataLoader
import numpy as np
from classifiers.neural_network import NeuralNetworkClassifier, NeuralNetworkNet
import torch

TRAIN_DATA_PATH = "data/train/one_hour"

EPOCHS = 5  ## TODO CHANGE TO 50
BATCH_SIZE=64

def load_classifier(num_of_inputs):
    net = NeuralNetworkNet(num_of_inputs)
    neural_network_classifier = NeuralNetworkClassifier(net)
    return {'model': neural_network_classifier, 'name': neural_network_classifier.name}


class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)

    def len(self):
        return len(self.X_data)


class NeuralNetworkPipeline():

    def __init__(self, target: str, train_data_path=TRAIN_DATA_PATH, ):
        self.train_data_path = train_data_path
        self.classifier = None
        self.model = None
        self.target = target
        self.read_train_validation()

    def read_train_validation(self):
        data = dp.read_train_data(TRAIN_DATA_PATH)
        features, targets, ids = dp.preprocess_data(data)

        X_train, X_validation, y_train, y_validation = dp.split_data(features, targets[self.target], test_size=0.2)

        self.X_train = X_train.to_numpy()
        self.X_validation = X_validation.to_numpy()

        self.y_train = y_train.to_numpy()
        self.y_validation = y_validation.to_numpy()

    def train_neural_network(self):

        self.standard_scaler = StandardScaler()
        self.X_train = self.standard_scaler.fit_transform(self.X_train)
        self.X_test = self.standard_scaler.transform(self.X_validation)

        ###########################
        # LOAD CLASSIFIER
        ###########################
        self.classifier = load_classifier(num_of_inputs=self.X_train.shape[1])

        self.model = self.classifier['model']
        print('start: {}'.format(self.classifier['name']))

        train_data = TrainData(torch.FloatTensor(self.X_train),
                               torch.from_numpy(self.y_train).view(-1, 1))  # problem with double solved
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

        ######################################################################
        # TRAIN MODE
        ######################################################################

        self.model.classifier.train()
        for epoch in range(EPOCHS):
            data_trained = 0
            data_length= self.X_train.shape[0]
            last_percentage = 0
            losses = []
            print(str.format('epoch {0}', epoch + 1))

            ### TRAIN AND VALIDATION SET

            for X_batch, y_batch in train_loader:
                loss = self.model.train(X_batch, y_batch)
                losses.append(loss)

                #####
                #OUTPUT
                ####
                data_trained += BATCH_SIZE
                current_percentage= data_trained / data_length * 100
                if current_percentage-last_percentage>10:
                    print('classifier: {0}, epoch status:{1:8.2f}%'.format(self.classifier['name'],current_percentage ))
                    last_percentage=current_percentage

            losses_np = np.array(losses)
            loss_mean = np.mean(losses_np)
            loss_std = np.std(losses_np)
            print(str.format('train loss: {0:1.3f} ± {1:1.3f}', loss_mean, loss_std))

        ##################################################
        # VALIDATION MODE
        ##################################################
        # model.classifier.eval()
        test_data = TestData(torch.FloatTensor(self.X_test))

        test_loader = DataLoader(dataset=test_data, batch_size=TestData.len(test_data))
        for X_batch in test_loader:
            y_pred = self.model.predict(X_batch)

        y_pred = torch.round(torch.sigmoid(y_pred))
        y_pred = y_pred.detach().numpy()
        y_pred = np.array(y_pred).flatten()
        return y_pred

    def perform_prediction(self, input_features):
        transformed_features = self.standard_scaler.transform(input_features)
        transformed_features = np.array(transformed_features)

        test_data = TestData(torch.FloatTensor(transformed_features))
        test_loader = DataLoader(dataset=test_data, batch_size=TestData.len(test_data))
        for X_batch in test_loader:
            y_pred = self.model.predict(X_batch)

        y_pred = torch.round(torch.sigmoid(y_pred))
        y_pred = y_pred.detach().numpy()
        y_pred = np.array(y_pred).flatten()
        return y_pred


def main():
    #############################
    # TARGET CAN BE ANY OF "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"
    #############################

    neural_network_pipeline = NeuralNetworkPipeline('reply_timestamp')
    y_pred = neural_network_pipeline.train_neural_network()

    result = Result(neural_network_pipeline.classifier['name'], neural_network_pipeline.model,
                    str(neural_network_pipeline.model.classifier.parameters()))
    result.calculate_and_store_metrics(neural_network_pipeline.y_validation, y_pred)
    result.store_result()
    utils.store_model(neural_network_pipeline.model, neural_network_pipeline.classifier['name'])


if __name__ == "__main__":
    main()
