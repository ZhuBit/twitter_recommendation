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
            nn.Linear(num_of_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
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
        self.lr=0.00001
        self.wd=0
        self.classifier = self.net.to(self.device)

        self.optimizer = optim.SGD(self.classifier.parameters(), lr=self.lr, momentum=0.9,
                                   weight_decay=self.wd)
        # TODO FIX THIS
        self.loss_fn = nn.MSELoss()

    def train(self, X, y)->float:

        tensor_input = torch.Tensor(X).to(self.device, dtype=torch.float)
        tensor_target = torch.Tensor(y).long().to(self.device,dtype=torch.float)

        self.classifier.train()

        # Sets gradients of all model parameters to zero
        self.optimizer.zero_grad()

        # Get predictions
        output = self.classifier(tensor_input)


        output= torch.flatten(output)

        tensor_target= torch.reshape(tensor_target, (len(tensor_target), 1))

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