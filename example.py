import torch
from torch import nn
import torch.fx
import functorch.compile
import random
import torch.functional as F
from compiler import compile

EPOCHS = 0

device = torch.device('mps')
torch.set_float32_matmul_precision('high')


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


model = SimpleModel()

if EPOCHS > 0:
    model = model.to(device)
    # model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    with open('iris.data') as file:
        dataset = [line.strip().split(',') for line in file]
    dataset = [(list(map(float, x[:-1])), classes.index(x[-1])) for x in dataset]
    dataset = [[*x, y] for x, y in dataset]
    random.shuffle(dataset)
    dataset = torch.tensor(dataset, device=device)

    for i in range(EPOCHS):
        for j in range(1000):
            data = torch.randint(len(dataset), (16,), device=device)
            data = dataset[data]
            x = data[:, :-1]
            y = model(x)
            yt = data[:, -1:]

            # One-hot encode yt using eye
            yt = torch.eye(3, device=device)[yt.long()][:, 0]

            optimizer.zero_grad()
            l = loss(y, yt)
            l.backward()
            optimizer.step()
        print(l.item())

    print('Results:')
    with torch.no_grad():
        print(model(torch.tensor([5.9, 3.0, 5.1, 1.8], device=device)))

    model = model.to('cpu')


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = torch.argmax(x, dim=-1)
        return x


model = ModelWrapper(model)

def model(x: torch.Tensor) -> torch.Tensor:
    x = x + x[:, None] + x[:, None, None]
    x = x + torch.tensor([[0], [1], [1], [1]])
    x = torch.sum(x, dim=0)
    return x

code = compile(model, torch.tensor([5.1, 3.5, 1.4, 0.2]))
print(code)
print('Expected output:', model(torch.tensor([5.1, 3.5, 1.4, 0.2])))