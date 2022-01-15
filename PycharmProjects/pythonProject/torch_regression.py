import torch
from torch.autograd import  Variable
import torch.nn.functional as F
import  matplotlib.pyplot as plt

test_ = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1) #1維數據->二維數據，因為torch處理二維
y = test_.pow(2) + 0.2*torch.rand(test_.size())

test_,y = Variable(test_),Variable(y)

class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(1,10,1)
print(net)

#優化
optimizer = torch.optim.SGD(net.parameters(),lr=0.5) #lr 學習效率
loss_fun = torch.nn.MSELoss() #均方差

for t in range(100):
    prediciton = net(test_)
    loss = loss_fun(prediciton,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
