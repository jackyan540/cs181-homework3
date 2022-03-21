# TODO - Complete Part2NeuralNetwork. Include class into your PDF submission!

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Part2NeuralNetwork(nn.Module):
    def __init__(self):
        super(Part2NeuralNetwork, self).__init__()
        ## TODO: Define your neural network layers here!
        ## Importantly, any modules which initialize weights should be initialized
        ## as member variables here. 
        ## Note: Keep track of the shape of your input tensors in the training
        ## and test sets, because it affects how you define your layers!
        ## You might find this resource helpful:
        ## https://towardsdatascience.com/pytorch-layer-dimensions-what-sizes-should-they-be-and-why-4265a41e01fd
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(3072,1000)
        self.layer2 = nn.Linear(1000,1000)
        self.layer3 = nn.Linear(1000,10)


    def forward(self, x):
        ## TODO: This is where you should apply the layers defined in the __init__
        ## method and the ReLU activation functions to the input x.
        #x = x.view(32,-1)
        #x = x.flatten()
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.relu(self.layer3(x))

# Display model architecture
model = Part2NeuralNetwork()
model.to(device)