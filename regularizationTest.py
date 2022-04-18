import torch
from torch.autograd import Variable
from torch.nn import functional as F


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(128, 32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.linear3 = torch.nn.Linear(16, 2)
    def forward(self, x):
        layer1_out = F.relu(self.linear1(x))
        layer2_out = F.relu(self.linear2(layer1_out))
        out = self.linear3(layer2_out)
        return out

batchsize = 4
lambda1, lambda2 = 0.5, 0.01

model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

inputs = Variable(torch.rand(batchsize, 128))
targets = Variable(torch.ones(batchsize).long())
l1_regularization, l2_regularization = torch.tensor(0), torch.tensor(0)

optimizer.zero_grad()
outputs = model(inputs)
cross_entropy_loss = F.cross_entropy(outputs, targets)
for param in model.parameters():
    l1_regularization += torch.norm(param, 1)**2
    l2_regularization += torch.norm(param, 2)**2

loss = cross_entropy_loss + l1_regularization + l2_regularization
loss.backward()
optimizer.step()



regularization_loss = 0
for param in model.parameters():
    regularization_loss += torch.sum(abs(param))

classify_loss = criterion(pred, target)
loss = classify_loss + lamda * regularization_loss

optimizer.zero_grad()
loss.backward()
optimizer.step()