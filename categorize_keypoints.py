import glob
import json
import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader

from collect_keypoints import NUM_KPTS

class KeypointsDataset(Dataset):
    def __init__(self, root_dir):
        X = []
        Y = []

        for filename in glob.glob("./Poses/**/*.json"):
            with open(filename, "r") as fp:
                data = json.load(fp)
            for keypoints in data["keypoints"]:
                points = []
                for x, y in keypoints:
                    points.extend([x, y])

                if len(points) > 0:
                    X.append(points)
                    Y.append([1.0 * float(data["tag"] == "Roubo"), ])

        X = numpy.array(X)
        Y = numpy.array(Y)

        self.X = torch.as_tensor(X).float()
        self.y = torch.as_tensor(Y).float()

        self.root_dir = root_dir

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(KeypointsDataset("root"), batch_size=batch_size)
test_dataloader = DataLoader(KeypointsDataset("root"), batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(NUM_KPTS * 2, 50),
            nn.Linear(50, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-1)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            print("PRED", pred.argmax(1))
            print("Y", y.flatten())
            correct += (pred.argmax(1) == y.flatten()).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")