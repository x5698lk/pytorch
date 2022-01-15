import torch
import torch.nn.functional as F
from torch.autograd import Variable
import  matplotlib.pyplot as plt

#fake data
x = torch.linspace(-5,5,200) # x data(tensor),shape=(100,1)
x = Variable(x)
x_np = x.data.numpy

y_relu = F.relu(x).data.numpy



