from sklearn.preprocessing import StandardScaler
import data_preprocessing as dp
import utils
from Result import Result
from torch.utils.data import Dataset, DataLoader
import numpy as np
from classifiers.neural_network import NeuralNetworkClassifier, NeuralNetworkNet
import torch

train_data_path = "data/one_hour"

EPOCHS=5 ## TODO CHANGE TO 50

def load_classifiers(num_of_inputs):
    net = NeuralNetworkNet(num_of_inputs)
    neural_network_classifier = NeuralNetworkClassifier(net)
    return [
        {'model': neural_network_classifier, 'name': neural_network_classifier.name}
    ]


class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


## test data
class testData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)

    def len(self):
        return len(self.X_data)


if __name__ == "__main__":
    data = dp.read_train_data(train_data_path)
    features, targets, ids = dp.preprocess_data(data)

    X_train, X_test, y_train, y_test = dp.split_data(features, targets['reply_timestamp'], test_size=0.2)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)

    classifiers = load_classifiers(X_train.shape[1])

    for classifier in classifiers:
        model = classifier['model']
        print('start: {}'.format(classifier['name']))

        train_data = trainData(torch.FloatTensor(X_train), torch.from_numpy(y_train).view(-1, 1)) #problem with double solved
        train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

        ######################################################################
        # TRAIN MODE
        ######################################################################

        model.classifier.train()
        for epoch in range(EPOCHS):
            losses = []
            print(str.format('epoch {0}', epoch + 1))

            ### TRAIN AND VALIDATION SET

            for X_batch, y_batch in train_loader:
                loss=model.train(X_batch, y_batch)
                losses.append(loss)

            losses_np = np.array(losses)
            loss_mean = np.mean(losses_np)
            loss_std = np.std(losses_np)
            print(str.format('train loss: {0:1.3f} ± {1:1.3f}', loss_mean, loss_std))

        ##################################################
        # TEST MODE
        ##################################################
        model.classifier.eval()
        test_data = testData(torch.FloatTensor(X_test))

        test_loader = DataLoader(dataset=test_data, batch_size=testData.len(test_data))
        for X_batch in test_loader:
            y_pred = model.predict(X_batch)


        y_pred = torch.round(torch.sigmoid(y_pred))
        y_pred = y_pred.detach().numpy()
        y_pred = np.array(y_pred).flatten()


        result = Result(classifier['name'], model, str(model.classifier.parameters()))
        result.calculate_and_store_metrics(y_test, y_pred)
        result.store_result()
        utils.store_model(model, classifier['name'])
