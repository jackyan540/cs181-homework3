class Problem3_7NeuralNetwork(nn.Module):
    def __init__(self):
        super(Problem3_7NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(3072,4000)
        self.layer2 = nn.Linear(4000,5000)
        self.layer3 = nn.Linear(5000,5000)
        self.layer4 = nn.Linear(5000,10)


    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return F.relu(self.layer4(x))

# Display model architecture
model = Problem3_7NeuralNetwork()
model.to(device)