#This is a basic ANN
def ANN_model():
    import torch as tor
    import torchvision as tv
    from torchvision import transforms, datasets
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    train = datasets.MNIST("", train = True, download = False, transform = transforms.Compose([transforms.ToTensor()]))

    test = datasets.MNIST("", train = False, download = False, transform = transforms.Compose([transforms.ToTensor()]))

    trainset = tor.utils.data.DataLoader(train, batch_size = 10, shuffle=True)
    testset = tor.utils.data.DataLoader(test, batch_size = 10, shuffle=True)

    input_size = 1*28*28
    layer1 = 64
    layer2 = 64
    layer3 = 64
    output_size = 10
    lr = 0.01
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_size, layer1)
            self.fc2 = nn.Linear(layer1, layer2)
            self.fc3 = nn.Linear(layer2, layer3)
            self.fc4 = nn.Linear(layer3, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return F.log_softmax(x, dim=1)

    net = Net()
    print(net)

    X = tor.rand((28,28))
    X = X.view(-1, 28*28) # -1 says will pass any size batch, just be ready



    optimizer = optim.Adam(net.parameters(), lr = lr)

    EPOCHS = 3
    for epoch in range(EPOCHS):
        for data in trainset:
            X, y = data #1 batch
            net.zero_grad()

            output = net(X.view(-1, 28*28))
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print(loss)

    correct = 0
    total = 0
    with tor.no_grad():
        for data in trainset:
            X, y = data
            output = net(X.view(-1, 28*28))
            for idx, i in enumerate(output):
                if tor.argmax(i) == y[idx]:
                    correct+=1
                total+=1

    print(correct/total)