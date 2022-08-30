import csvloader
import torch
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm

torch.set_num_interop_threads(16)
torch.set_num_threads(16)

tensors = csvloader.load("1/day_head_circumference.csv")
day_train = tensors[:, 0:1]
circumference_train = tensors[:, 1:2]


class LinRegModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinRegModel()
optimizer = torch.optim.SGD([model.W, model.b], 0.000_000_1)
for epoch in tqdm(range(1_000_000)):
    model.loss(day_train, circumference_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print(
    "W = %s, b = %s, loss = %s"
    % (model.W, model.b, model.loss(day_train, circumference_train))
)

steps = 100
day = torch.linspace(torch.min(day_train), torch.max(day_train), steps).reshape(-1, 1)
circ = model.f(day)

plt.plot(
    day.detach().numpy(),
    circ.detach().numpy(),
    "",
    color="violet",
    label="$(x^{(i)},y^{(i)})$",
)
plt.xlabel("length")
plt.ylabel("weight")
plt.scatter(day_train, circumference_train, s=2)

x = torch.tensor([[torch.min(day_train)], [torch.max(day_train)]])
plt.legend()
plt.savefig("1/c.png")
