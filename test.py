import numpy
import torch

print("torch.__version__=" + torch.__version__)
print("torch.version.cuda=", end="")
print(torch.version.cuda)
print("torch.cuda.is_available()=", end="")
print(torch.cuda.is_available())

data = numpy.load('data/ECG_train_data.npy')
print(data)