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

batch_size = 100
train_data = MNIST_Dataset(train_image)

train_data_loader = DataLoader(train_data, batch_size = batch_size,
                               shuffle = True)

test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)


#arguments
num_stochastic_layers = 1
model = 'IWAE'
num_samples = 5
#############################################


vae = IWAE_1(20, 10, 784, 1)

vae.double()

vae.to(device)

    
#optimizer = optim.Adam(vae.parameters(), weight_decay = 0, amsgrad = True)
num_epoches = 50
train_loss_epoch = []

total_step = 0
temp = 1.0
for epoch in range(num_epoches):
    #learning_rate
    if  epoch == 0:
        lr = 1e-3
    elif epoch>0 and epoch<=3:
        lr = 1e-3 * np.power(10, -1/7)
    elif epoch>3 and epoch<=(3+3**2):
        lr = 1e-3 * np.power(10, -2/7)
    elif epoch>3+3**2 and epoch<=(3+3**2+3**3):
        lr = 1e-3 * np.power(10, -3/7)
    elif epoch>3+3**2+3**3 and epoch<=(3+3**2+3**3+3**4):
        lr = 1e-3 * np.power(10, -4/7)
    elif epoch>3+3**2+3**3+3**4 and epoch<=(3+3**2+3**3+3**4+3**5):
        lr = 1e-3 * np.power(10, -5/7)
    elif epoch>3+3**2+3**3+3**4+3**5 and epoch<=(3+3**2+3**3+3**4+3**5+3**6):
        lr = 1e-3 * np.power(10, -6/7)
    elif epoch>3+3**2+3**3+3**4+3**5+3**6 and epoch<=(3+3**2+3**3+3**4+3**5+3**6+3**7):
        lr = 1e-3 * np.power(10, -7/7)
        
        
    optimizer = optim.Adam(vae.parameters(), lr = lr, weight_decay = 0, amsgrad = False, eps=1e-04)
    running_loss = []    
    
    for idx, data in enumerate(train_data_loader):
        total_step+=1
        data = data.double()
        inputs = Variable(data).to(device)
        if model == "IWAE":
            inputs = inputs.unsqueeze_(0)
            inputs = inputs.expand(num_samples, batch_size, 784)
        elif model == "VAE":
            inputs = inputs.repeat(num_samples, 1)
            inputs = inputs.unsqueeze_(0)
            inputs = inputs.expand(1, batch_size*num_samples, 784)
            
        optimizer.zero_grad()
        loss = vae.train_loss(inputs, temp)
        loss.backward()
        optimizer.step()
        if (total_step+1)%100==0:
            temp = np.maximum(temp*np.exp(-0.00003*total_step),0.5)
            
        if (idx+1) % 100 == 0:
            
            print(("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}").format(epoch, idx, loss.item()), flush = True)
        running_loss.append(loss.item())        
    train_loss_epoch.append(np.mean(running_loss))

    if (epoch + 1) % 50 == 0:
        torch.save(vae.state_dict(),
                   ("{}_layers_{}_k_{}_epoch_{}_Gumbel.model")
                   .format(model, num_stochastic_layers,
                           num_samples, epoch))