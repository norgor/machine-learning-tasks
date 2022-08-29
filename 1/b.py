import csvloader
import torch
import matplotlib.pyplot as plt

tensors = csvloader.load("1/day_length_weight.csv")
y_train = tensors[:, :1]
x_train = tensors[:, 1:]


class LinRegModel:
    def __init__(self):
        self.W = torch.tensor([[0.0, 0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0, 0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinRegModel()
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(1_000_000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

plt.plot(x_train, y_train, "o", label="$(x^{(i)},y^{(i)})$")
plt.xlabel("length")
plt.ylabel("weight")
plt.scatter(x_train, y_train, s=2)

x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x, model.f(x).detach(), label="$\\hat y = f(x) = xW+b$")
plt.legend()
plt.savefig("1/b.png")
