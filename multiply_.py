import numpy as np
import torch
import math

def sigmoid(v):
    return 1 / (1 + torch.exp(-10*v))

x1=np.array([0.2, 0.5, 0.1])
w1=np.array([[0.2, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0.2]])
w2=np.array([[0.5, 0.5, 0], [-0.5, 0.5, -0.5]])
w3=np.array([[0.1, 0, 0, 0], [-0.9, -0.5, 0.9, 0], [0, 0, -0.1, 0.5]])
w4=np.array([[0.6, -0.3, 0.1], [0.3, -0.9, -0.1]])

x2=np.dot(w1,x1)
x2_ac=sigmoid(torch.FloatTensor(x2))
print(x2_ac)

x3=np.dot(w2, x2_ac)
x3_ac=sigmoid(torch.FloatTensor(x3))
print(x3_ac)

x3ac_x2ac=np.array([0.9997, 0.8320, 0.0243, 0.7685])
x4=np.dot(w3, x3ac_x2ac)
x4_ac=sigmoid(torch.FloatTensor(x4))
print(x4_ac)

ypred=np.dot(w4,x4_ac)
print(ypred)
ypred_ac=sigmoid(torch.FloatTensor(ypred))
print(ypred_ac)

# m = torch.sigmoid(torch.FloatTensor(x2*10))
# print(m)
# print(1 / (1 + math.exp(-10*0.14)))

print(
    (
    (0.9953-1)*(10*0.9953*(1-0.9953))*0.6 + \
    (0.7711-0)*(10*0.7711*(1-0.7711))*0.3
    ) * (10*0.7310*(1-0.7310)) * 0.9997
)

delta_1=(0.9953-1)*(10*0.9953*(1-0.9953))
delta_2=(0.7711-0)*(10*0.7711*(1-0.7711))
print(delta_1)
print(delta_2)
