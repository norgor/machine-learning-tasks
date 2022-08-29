import csvloader
import torch
import matplotlib.pyplot as plt

tensors = csvloader.load("1/length_weight.csv")
length_train = tensors[:, 0:1]
weight_train = tensors[:, 1:2]


class LinRegModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinRegModel()
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(1_000_000):
    model.loss(length_train, weight_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print(
    "W = %s, b = %s, loss = %s"
    % (model.W, model.b, model.loss(length_train, weight_train))
)

plt.plot(length_train, weight_train, "o", label="$(x^{(i)},y^{(i)})$")
plt.xlabel("length")
plt.ylabel("weight")
plt.scatter(length_train, weight_train, s=2)

x = torch.tensor([[torch.min(length_train)], [torch.max(length_train)]])
plt.plot(x, model.f(x).detach(), label="$\\hat y = f(x) = xW+b$")
plt.legend()
plt.savefig("1/a.png")
