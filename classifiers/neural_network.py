from classifiers.base_classifier import BaseClassifier
import os
import torch
from torch import nn
import torch.optim as optim


class NeuralNetworkNet(nn.Module):
    def __init__(self, num_of_inputs):
        super(NeuralNetworkNet, self).__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(num_of_inputs, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(0.1),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 1),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.net(x)
        return logits

############################################################

############################################################


class NeuralNetworkClassifier(BaseClassifier):
    def __init__(self, net: nn.Module):
        super(NeuralNetworkClassifier, self,).__init__("NeuralNetwork")
        self.net = net
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_type)
        self.lr=0.001
        self.wd=0
        self.classifier = self.net.to(self.device)

        self.optimizer = optim.SGD(self.classifier.parameters(), lr=self.lr, momentum=0.0,
                                   weight_decay=self.wd)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def train(self, X:torch.FloatTensor, y:torch.FloatTensor)->float:

        tensor_input = X
        y_numpy = (torch.flatten(y)).detach().numpy()
        y_numpy = y_numpy.astype(float)
        tensor_target = torch.FloatTensor(y_numpy)

        self.classifier.train()

        # Sets gradients of all model parameters to zero
        self.optimizer.zero_grad()

        # Get predictions
        output = self.classifier(tensor_input)
        output= torch.flatten(output)


        # Compute the loss
        loss = self.loss_fn(output, tensor_target)

        # Compute gradients
        loss.backward()

        # Take an optimization step (parameter update)
        self.optimizer.step()

        return loss.item()

    def predict(self, X_test):
        # Pass the network's predictions through a nn.Softmax layer to obtain softmax class scores
        # Make sure to set the network to eval() mode
        # See above comments on CPU/GPU
        tensor_input = torch.Tensor(X_test).to(self.device)

        # Sets the module in evaluation mode
        self.classifier.eval()
        # Get predictions
        output = self.classifier(tensor_input)
        return output