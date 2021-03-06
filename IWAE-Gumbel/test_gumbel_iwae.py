import numpy as np
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn  
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
sys.path.append("../model/")

from IWAE_Gumbel import *
from sys import exit

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

## read data
with open('data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)
    
train_image = data['train_image']
train_label = data['train_label']
test_image = data['test_image']
test_label = data['test_label']

batch_size = 1
train_data = MNIST_Dataset(train_image)

train_data_loader = DataLoader(train_data, batch_size = batch_size)

test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)

num_stochastic_layers = 1
model = 'IWAE'
num_samples = 5
epoch = 49  #last training epoch
    

vae = IWAE_1(20, 10, 784, 1)


vae.double()
vae.to(device)
model_file_name = ("{}_layers_{}_k_{}_epoch_{}_Gumbel.model").format(
                       model, num_stochastic_layers,
                       num_samples,epoch)
vae.load_state_dict(torch.load(model_file_name))

tot_loss = 0
tot_size = 0
for idx, data in enumerate(test_data_loader):
    print(idx)
    data = data.double()
    with torch.no_grad():
        inputs = Variable(data).to(device)
        inputs = inputs.expand(1000, batch_size, 784)    
        loss = vae.test_loss(inputs,0.05)

        size = inputs.size()[0]
        tot_size += size
        tot_loss += loss.item() * size

print(model_file_name, end = ",")        
print("Average loss: {:.2f}".format(tot_loss/tot_size))