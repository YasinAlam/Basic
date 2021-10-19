def CNN_model(REBUILD_DATA=False):
    import torch as tor
    import torchvision as tv
    from torchvision import transforms, datasets
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    import os
    import cv2
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    #REBUILD_DATA = False

    class DogsVSCats():
        IMG_SIZE = 50
        CATS = "kagglecatsanddogs_3367a/PetImages/Cat"
        DOGS = "kagglecatsanddogs_3367a/PetImages/Dog"
        LABELS = {CATS: 0, DOGS: 1}
        training_data = []
        catcount = 0
        dogcount = 0
        def make_training_data(self):
            for label in self.LABELS:
                print(label)
                for f in tqdm(os.listdir(label)):
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])

                        if label == self.CATS:
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount +=1
                    except Exception as e:
                        pass
            np.random.shuffle(self.training_data)
            np.save("training_data.npy", self.training_data)
            print("Cats: {}, Dogs: {},".format(self.catcount, self.dogcount))

    if REBUILD_DATA:
        dvc = DogsVSCats()
        dvc.make_training_data()

    training_data = np.load("training_data.npy", allow_pickle=True)
    #plt.imshow(training_data[1][0], cmap = 'gray')
    #plt.show()

    kernal = 5
    conchan1 = 32
    conchan2 = 64
    conchan3 = 128
    fc1 = 512
    output_size = 2
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, conchan1, kernal)
            self.conv2 = nn.Conv2d(conchan1, conchan2, kernal, )
            self.conv3 = nn.Conv2d(conchan2, conchan3, kernal)

            x = tor.randn(50, 50).view(-1,1,50,50)
            self._to_linear = None
            self.convs(x)

            self.fc1 = nn.Linear(self._to_linear, fc1 )
            self.fc2 = nn.Linear(fc1, output_size)

        def convs(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
            x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

            if self._to_linear is None:
                self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            return x

        def forward(self, x):
            x = self.convs(x)
            x = x.view(-1, self._to_linear)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.softmax(x, dim = 1)

    net = Net()
    lr = 0.001
    optimizer = optim.Adam(net.parameters(), lr = lr)
    loss_function = nn.MSELoss()

    X = tor.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
    X /= 255.0
    y = tor.Tensor([i[1] for i in training_data])

    VAL_PCT= 0.1
    val_size = int(len(X)*VAL_PCT)
    print(val_size)

    train_X = X[:-val_size]
    train_y = y[:-val_size]

    test_X = X[-val_size:]
    test_y = y[-val_size:]

    BATCH_SIZE = 100

    EPOCHS = 1

    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
            batch_y = train_y[i:i+BATCH_SIZE]

            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
    print(loss)

    correct = 0
    total = 0
    with tor.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = tor.argmax(test_y[i])
            net_out = net(test_X[i].view(-1,1,50,50))[0]

            pred = tor.argmax(net_out)
            if pred == real_class:
                correct+=1
            total+=1
    print(correct/total)