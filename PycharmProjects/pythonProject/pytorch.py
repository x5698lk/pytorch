import torch
import numpy as np


np_data = np.arange(8).reshape((2,4))
torch_data = torch.from_numpy((np_data))
tensor2array = torch_data.numpy()

data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)
data = np.array(data)

#numpy->tensor->numpy
print(
    '\nnumpy:',np_data,
    '\ntorch',torch_data,
    '\ntensor2array',tensor2array,
    '\n\n',
#矩陣運算
    '\nnumpy:',np.matmul(data, data),
    '\ntorch',torch.mm(tensor,tensor),
    '\nnumpy:',data.dot(data),
)
