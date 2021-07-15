from sklearn.preprocessing import StandardScaler
import utils
from Result import Result
from torch.utils.data import Dataset, DataLoader
import numpy as np
from classifiers.neural_network import NeuralNetworkClassifier, NeuralNetworkNet
import torch

from data_preprocessing import DataPreprocessing
from data_preprocessing import split_data

TRAIN_DATA_PATH = "data/train/one_hour"
TEST_DATA_PATH = "data/validation/one_hour"
TARGET = "reply_timestamp"
# "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"

EPOCHS = 1  ## TODO CHANGE TO 50
BATCH_SIZE = 64

MODE = "TEST"  # ["TRAIN_TEST","TEST"]
SAVING_MODEL = MODE == "TRAIN_TEST"


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

        self.standard_scaler = StandardScaler()

        self.target = target


        self.read_train_validation()

        ###################
        # FIT SCALER
        ###################
        self.X_train = self.standard_scaler.fit_transform(self.X_train)
        self.X_test = self.standard_scaler.transform(self.X_validation)

        ###########################
        # LOAD CLASSIFIER
        ###########################
        self.classifier = load_classifier(num_of_inputs=self.X_train.shape[1])
        self.model = self.classifier['model']


    def read_train_validation(self):
        data_preprocessing = DataPreprocessing(self.train_data_path)
        features, targets = data_preprocessing.get_processed_data()
        X_train, X_validation, y_train, y_validation = split_data(features, targets['reply_timestamp'], test_size=0.2)

        self.X_train = X_train.to_numpy()
        self.X_validation = X_validation.to_numpy()

        self.y_train = y_train.to_numpy()
        self.y_validation = y_validation.to_numpy()



    def train_neural_network(self):




        print('start: {}'.format(self.classifier['name']))

        train_data = TrainData(torch.FloatTensor(self.X_train),
                               torch.from_numpy(self.y_train).view(-1, 1))  # problem with double solved
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

        ######################################################################
        # TRAIN MODE
        ######################################################################
        best_accuracy = 0

        result = Result(self.classifier['name'], self.model,
                        str(self.model.classifier.parameters()))
        self.model.classifier.train()
        for epoch in range(EPOCHS):
            data_trained = 0
            data_length = self.X_train.shape[0]
            last_percentage = 0
            losses = []
            print(str.format('epoch {0}', epoch + 1))

            ### TRAIN AND VALIDATION SET

            for X_batch, y_batch in train_loader:
                loss = self.model.train(X_batch, y_batch)
                losses.append(loss)

                #####
                # OUTPUT
                ####
                data_trained += BATCH_SIZE
                current_percentage = data_trained / data_length * 100
                if current_percentage - last_percentage > 10:
                    print('classifier: {0}, epoch status:{1:8.2f}%'.format(self.classifier['name'], current_percentage))
                    last_percentage = current_percentage

            losses_np = np.array(losses)
            loss_mean = np.mean(losses_np)
            loss_std = np.std(losses_np)
            print(str.format('train loss: {0:1.3f} ± {1:1.3f}', loss_mean, loss_std))

            ##################################################
            # VALIDATION MODE
            ##################################################
            self.model.classifier.eval()
            test_data = TestData(torch.FloatTensor(self.X_test))
            test_loader = DataLoader(dataset=test_data, batch_size=TestData.len(test_data))
            for X_batch in test_loader:
                y_pred = self.model.predict(X_batch)

            y_pred = torch.round(torch.sigmoid(y_pred))
            y_pred = y_pred.detach().numpy()
            y_pred = np.array(y_pred).flatten()

            correct = np.sum(y_pred == self.y_validation)

            # Update the total number of samples
            total_number = len(y_pred)

            # Calculate the accuracy
            accuracy = correct / total_number
            print(str.format('epoch {0}, validation accuracy:{1}', epoch + 1, accuracy))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if SAVING_MODEL:
                    torch.save(self.model.net.state_dict(), 'best_model_{}.pt'.format(TARGET))

                # result.calculate_and_store_metrics(self.y_validation, y_pred)
                # result.store_result()
                # utils.store_model(neural_network_pipeline.model, neural_network_pipeline.classifier['name'])

    def perform_prediction(self, X: np.array):
        X_transformed = self.standard_scaler.transform(X)
        X_transformed = np.array(X_transformed)

        test_data = TestData(torch.FloatTensor(X_transformed))
        test_loader = DataLoader(dataset=test_data, batch_size=TestData.len(test_data))
        for X_batch in test_loader:
            y_pred = self.model.predict(X_batch)

        y_pred = torch.round(torch.sigmoid(y_pred))
        y_pred = y_pred.detach().numpy()
        y_pred = np.array(y_pred).flatten()
        return y_pred

    def load_model(self, PATH, num_of_inputs):
        self.model.net.load_state_dict(torch.load(PATH))


def main():
    #############################
    # TARGET CAN BE ANY OF "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"
    #############################

    if MODE == "TRAIN_TEST":
        print("Loading datasets...")
        neural_network_pipeline = NeuralNetworkPipeline(TARGET)
        print("Datasets loaded.")
        print("Starting training...")

        neural_network_pipeline.train_neural_network()

        print("Training done!")
    #######################################
    # TEST
    #######################################

    data_preprocessing = DataPreprocessing(TEST_DATA_PATH)
    X_test, y_test = data_preprocessing.get_processed_data()
    y_test = y_test[TARGET]
    y_test = np.array(y_test)
    y_test = y_test.flatten()
    # print(len(X_test))

    if MODE == "TEST":
        print("Loading datasets...")
        neural_network_pipeline = NeuralNetworkPipeline(TARGET)
        print("Datasets loaded.")

        print("Loading model...")
        neural_network_pipeline.load_model('best_model_{}.pt'.format(TARGET), X_test.shape[1])
        print("Model loaded.")

    y_pred = neural_network_pipeline.perform_prediction(X_test)
    # print(y_pred, len(y_pred))
    # print(y_test, len(y_test))

    result = Result(neural_network_pipeline.classifier['name'], neural_network_pipeline.model,
                    str(neural_network_pipeline.model.classifier.parameters()))
    result.calculate_and_store_metrics(y_test, y_pred)
    result.store_result()
    # utils.store_model(neural_network_pipeline.model, neural_network_pipeline.classifier['name'])


if __name__ == "__main__":
    main()
