import glob
import json
import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from collect_keypoints import NUM_KPTS


class KeypointsDataset(Dataset):
    def __init__(self, pose_folder):
        X = []
        Y = []
        for filename in glob.glob(f"./Poses/{pose_folder}/*.json"):
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

        self.root_dir = "root"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RobberyKeypoints(KeypointsDataset):
    def __init__(self):
        KeypointsDataset.__init__(self, "Roubo")

class NoRobberyKeypoints(KeypointsDataset):
    def __init__(self):
        KeypointsDataset.__init__(self, "NaoRoubo")

batch_size = 64

r_kpts = RobberyKeypoints()
nr_kpts = NoRobberyKeypoints()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

KPS_SIZE = NUM_KPTS * 2
KPS_PWSET = KPS_SIZE ** 2

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(KPS_SIZE, KPS_PWSET),
            nn.ReLU(),
            nn.Linear(KPS_PWSET, KPS_PWSET),
            nn.ReLU(),
            nn.Linear(KPS_PWSET, KPS_PWSET),
            nn.ReLU(),
            nn.Linear(KPS_PWSET, KPS_PWSET),
            nn.ReLU(),
            nn.Linear(KPS_PWSET, KPS_PWSET),
            nn.Linear(KPS_PWSET, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits


model = NeuralNetwork().to(device)
loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)


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

epochs = 150

t_tp = 0
t_fp = 0
t_tn = 0
t_fn = 0
t_precision = 0
t_accuracy = 0
t_recall = 0
t_f1_score = 0
t_sensitivity = 0
t_specificity = 0
t_losses = numpy.array([0.0 for _ in range(epochs)])
t_correct = numpy.array([0.0 for _ in range(epochs)])

ALGO_EXECUTIONS = 20
for k in range(ALGO_EXECUTIONS):
    train_r_kpts, test_r_kpts = random_split(r_kpts, [int(0.8 * len(r_kpts)), len(r_kpts) - int(0.8 * len(r_kpts))])
    train_nr_kpts, test_nr_kpts = random_split(nr_kpts, [int(0.8 * len(nr_kpts)), len(nr_kpts) - int(0.8 * len(nr_kpts))])
    train_data = train_nr_kpts + train_r_kpts
    test_data = test_r_kpts + test_nr_kpts
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    print(f"Running iteration {k}")
    losses = []
    accs = []

    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer)
        correct, loss = test(test_dataloader, model, loss_fn)
        t_losses[t] = t_losses[t] + loss
        t_correct[t] = t_correct[t] + correct

    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            output = model(inputs.to(device))
            output = torch.round(output.flatten()).cpu().numpy()
            y_pred.extend(output)

            labels = labels.flatten().cpu().numpy()
            y_true.extend(labels)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    t_tn += tn
    t_fp += fp
    t_fn += fn
    t_tp += tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    t_precision += precision
    t_accuracy += (tp + tn) / (tp + tn + fp + fn)
    t_recall += recall
    t_f1_score +=  2 * (precision * recall) / (precision + recall)
    t_sensitivity += tp / (tp + fn)
    t_specificity += tn / (tn + fp)

losses = t_losses / ALGO_EXECUTIONS
correct = t_correct / ALGO_EXECUTIONS

tp = t_tp / ALGO_EXECUTIONS
tn = t_tn / ALGO_EXECUTIONS
fp = t_fp / ALGO_EXECUTIONS
fn = t_fn / ALGO_EXECUTIONS
precision = t_precision / ALGO_EXECUTIONS
accuracy = t_accuracy / ALGO_EXECUTIONS
recall = t_recall / ALGO_EXECUTIONS
f1_score = t_f1_score / ALGO_EXECUTIONS
sensitivity = t_sensitivity / ALGO_EXECUTIONS
specificity = t_specificity / ALGO_EXECUTIONS

print(f"TP: {tp:.2f}, TN: {tn:.2f}, FP: {fp:.2f}, FN: {fn:.2f}")
print(f"Precision: {precision:.2f}, Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")
print(f"Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}")

sns.set_theme(style="darkgrid")

plt.xlabel("Epoch")
plt.ylabel("Loss - Binary Cross Entropy")
sns.lineplot(range(len(losses)), losses)

plt.savefig("img.png")