import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)


t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

v_out.backward()

print(tensor)
print(variable)
print('\n\n')
print(t_out)
print(v_out)
print('\n\n')
print(variable.grad)
print(variable.data)
print(variable.data.numpy())