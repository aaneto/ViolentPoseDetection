import glob
import json
import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import minmax_scale

from collect_keypoints import NUM_KPTS

class KeypointsDataset(Dataset):
    def __init__(self, root_dir, training):
        X = []
        Y = []
        i = -1
        for filename in glob.glob("./Poses/**/*.json"):
            i += 1
            if i % 2 == 0 and training:
                continue
            elif i % 2 != 0 and not training:
                continue

            with open(filename, "r") as fp:
                data = json.load(fp)
            for keypoints in data["keypoints"]:
                points = []
                for x, y in keypoints:
                    points.extend([x, y])

                if len(points) > 0:
                    X.append(points)
                    Y.append([1.0 * float(data["tag"] == "Roubo"), ])

        X = minmax_scale(numpy.array(X))
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
train_dataloader = DataLoader(KeypointsDataset("root", True), batch_size=batch_size)
test_dataloader = DataLoader(KeypointsDataset("root", False), batch_size=batch_size)

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
        self.stack = nn.Sequential(
            nn.Linear(NUM_KPTS * 2, 300),
            nn.Sigmoid(),
            nn.Linear(300, 30),
            nn.Linear(30, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            correct += (torch.round(pred) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    return correct, test_loss

epochs = 200
for t in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer)
    correct, loss = test(test_dataloader, model, loss_fn)

print(f"Accuracy after {epochs} epochs: {int(100.0 * correct)}%")

with torch.no_grad():
    x, y  = next(iter(test_dataloader))
    x, y = x.to(device), y.to(device)
    pred = model(x)
    print(torch.round(pred.flatten()), y.flatten())