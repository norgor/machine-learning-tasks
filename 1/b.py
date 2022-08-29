import csvloader
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('WebAgg')

tensors = csvloader.load("1/day_length_weight.csv")
y_train = tensors[:, :1]
x_train = tensors[:, 1:]


class LinRegModel:
    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

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

axes = plt.axes(projection="3d")
axes.scatter3D(x_train[:, 0:1], x_train[:, 1:2], y_train)
# plt.plot(x_train, y_train, "o", label="$(x^{(i)},y^{(i)})$")
axes.set_xlabel("length")
axes.set_ylabel("weight")
axes.set_zlabel("age (days)")

steps = 2
x = torch.linspace(torch.min(x_train[:, 0:1]), torch.max(x_train[:, 0:1]), steps)
y = torch.linspace(torch.min(x_train[:, 1:2]), torch.max(x_train[:, 1:2]), steps)
a, b = torch.meshgrid(x, y)
c = torch.stack((a.reshape(-1, 1), b.reshape(-1, 1)), 1).reshape(-1, 2)
z = model.f(c).reshape(1, -1)[0]

axes.plot_trisurf(
    a.detach().numpy().reshape(1, -1)[0], 
    b.detach().numpy().reshape(1, -1)[0], 
    z.detach().numpy(), 
    alpha=0.7,
    color="violet"
)

# x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
# plt.plot(x, model.f(x).detach(), label="$\\hat y = f(x) = xW+b$")
plt.savefig("1/b.png")
plt.show()
