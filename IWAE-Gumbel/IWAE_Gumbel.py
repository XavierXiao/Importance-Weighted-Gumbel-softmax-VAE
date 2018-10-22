import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class MNIST_Dataset(Dataset):    
    def __init__(self, image):
        super(MNIST_Dataset).__init__()
        self.image = image
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, idx):
        return np.random.binomial(1, self.image[idx, :]).astype('float32')
    '''
    ramdomly binarized MNIST
    '''


def sample_gumbel(shape, epsilon=1e-20): 
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape).to('cuda')
        U = U.double()
        return -Variable(torch.log(-torch.log(U + epsilon) + epsilon)).double()

def gumbel_softmax_sample(logits, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits.to('cuda').double()+ sample_gumbel(logits.size())
    return F.softmax( y / temperature, dim=-1)
    
#if hard=True, temp fixed at 1 during training
def gumbel_softmax(logits, temperature, hard=False):
    u = gumbel_softmax_sample(logits, temperature)
    if hard:
        shape = u.size()
        _, ind = u.max(dim=-1)
        u_hard = torch.zeros_like(u).view(-1, shape[-1])
        u_hard.scatter_(1, ind.view(-1, 1), 1)
        u_hard = u_hard.view(*shape)
        u = (u_hard - u).detach() + u
    return u


 
class Encoder(nn.Module):
    '''
    encoder
    '''
    def __init__(self, input_dim, hidden_dim, dim_h1, categorical_dim):
        '''
        input_sim = 784
        hidden_dim = 200
        output_dim = 50
        '''
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dim_h1 = dim_h1
        self.categorical_dim = categorical_dim
        
        self.transform = nn.Sequential(nn.Linear(input_dim, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, 256),
                                       nn.ReLU())
        self.to_hiddden = nn.Linear(256, dim_h1*categorical_dim)
        #self.softmax = nn.Softmax(dim=-1)
        
    
    def forward(self, x): #according to Jang's code, use softmax after linear layer
        out = self.transform(x)
        out = self.to_hiddden(out)
        '''
        This return the unnormalized logits, out is of size 
        [num_sample, batch_size, N*K], need to reshape for softmax
        '''
        out = out.view(out.size()[0],out.size()[1],self.dim_h1, self.categorical_dim)
        #out = self.softmax(out)
        #out = out.view(out.size()[0],out.size()[1],self.dim_h1*self.categorical_dim)
        
        return out
'''
#################################################################################################
'''

class IWAE_1(nn.Module):
    def __init__(self, dim_h1, categorical_dim, dim_image_vars,temp):
        super(IWAE_1, self).__init__()
        self.dim_h1 = dim_h1
        self.dim_image_vars = dim_image_vars
        self.categorical_dim = categorical_dim

        ## encoder
        self.encoder = Encoder(dim_image_vars, 200, dim_h1, categorical_dim)
        #here, the tensor is of size [num_sample, batch_numer, K*N]
        
        ## decoder
        self.decoder =  nn.Sequential(nn.Linear(dim_h1*categorical_dim, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, dim_image_vars),
                                        nn.Sigmoid())
        
        self.softmax = nn.Softmax(dim=-1)
       


#    def encoder(self, x):
#        logits = self.encoder_h1(x)
#        #logits = logits.view(x.size()[0],x.size()[1],self.dim_h1, self.categorical_dim)
#        '''
#        the logits is already the size [num_sample, batch_size, N, K], it is unnormalized, ready
#        to sent to the sampler
#        '''
#        
#        return logits
#    
#    def decoder(self, h1):
#        '''
#        h1 should be [num_sample, batch_size, N* K]
#        p should be [num_sample, batch_size, 784]
#        '''
#        p = self.decoder_x(h1)
#        return p
#    
    def forward(self, x, temp):
        logits = self.encoder(x)
        #logits [num_sample, batch_size, N,K], which fits the input dim of gumbel_softmax
        z = gumbel_softmax(logits,temp)
        #Below is to reshape to [num_sample, batch_size, N*K] to fit into decoder
        z = z.view(z.size()[0],z.size()[1],z.size()[2]*z.size()[3])
        z = z.to('cuda').double()
        p = self.decoder(z)
        return p,z

    def train_loss(self, inputs,temp):
        logits = self.encoder(inputs)
        '''
        logits is of size [num_sample,batch_size, N,K]
        '''
        q_y = self.softmax(logits) #softmax of logits, only used in computing loss
        
        z_unreshape = gumbel_softmax(logits,temp)
        z_unreshape =  z_unreshape.to('cuda').double()
        '''
        z_unreshape is of size [num_sample,batch_size, N,K]
        '''
        
        z = z_unreshape.view(z_unreshape.size()[0],z_unreshape.size()[1],z_unreshape.size()[2]*z_unreshape.size()[3])
        #z = z.to('cuda').double()
        '''
        z is of size [num_sample,batch_size, N*K] in order to feed in decoder
        '''
        p = self.decoder(z)
        #p is of size [num_sample, batch_size, 784]
        
        
        #log_PxGh1 = torch.sum(inputs*torch.log(p) + (1-inputs)*torch.log(1-p), -1)
        log_PxGh1 = -torch.sum(F.binary_cross_entropy(p.view(p.size()[0],p.size()[1],784), inputs.view(p.size()[0],p.size()[1],784), reduction = 'none'),-1)
        #log_PxGh1 = -torch.sum(F.binary_cross_entropy(p.view(-1,784), inputs.view(-1,784), reduction = 'none'),-1)
        '''
        log likelihod for the decoder, which is a log likelihood over bernoulli
        this has size [num_sample,batch_size]
        '''
        ###################################################################################
        
        log_q_y = torch.log(q_y + 1e-20)
        '''
        log_posterior
        this line has size [k,batch_size, N,K]
        '''
        
        g = Variable(torch.log(torch.Tensor([1.0/self.categorical_dim])).to('cuda'))
        g = g.double()
        '''
        log_prior is just uniform among K categories
        
        '''
        #second_term = torch.sum(q_y*(g-log_q_y),dim=[2,3])
        second_term = torch.sum(z_unreshape*(g-log_q_y),dim=[2,3])
        '''
        TRY BOTH logits*(g-log_logits) and z_unreshape*(g-log_logits), second one makes more sense,
        but author use the first one.
        size [k,batch_size,N]
        '''

        log_weight = log_PxGh1 + second_term
        '''
        matrix of log(w_i)
        this has size [k,batch_size]
        '''
        
        log_weight = log_weight - torch.max(log_weight, 0)[0]
        '''
        normalize to prevent overflow
        maximum w's for each batch element, where maximum is taking over k samples from posterior
        
        Note: For plian version of VAE, this is identically 0
        '''
        
        weight = torch.exp(log_weight)
        '''
        exponential the log back to get w_i's
        Note: For plian version of VAE, this is identically 1
        '''
        
        weight = weight / torch.sum(weight, 0)
        '''
        \tilda(w_i)
        Note: For plian version of VAE, this is identically 1
        '''
       
        weight = Variable(weight.data, requires_grad = False)
        '''
        stop gradient on \tilda(w)
        '''
        
        loss = -torch.mean(torch.sum(weight * (log_PxGh1 + second_term), 0))
        #loss = -torch.mean(torch.sum((1/inputs.size()[0]) * (log_PxGh1 + second_term), 0))
        return loss

    def test_loss(self, inputs, temp):
        #TO DO, modify test_loss as train_loss
        logits = self.encoder(inputs)
        q_y = self.softmax(logits) #softmax of logits, only used in computing loss
        
        z_unreshape = gumbel_softmax(logits,temp)
        z_unreshape =  z_unreshape.to('cuda').double()
        z = z_unreshape.view(z_unreshape.size()[0],z_unreshape.size()[1],z_unreshape.size()[2]*z_unreshape.size()[3])
        #z = z.to('cuda').double()
        p = self.decoder(z)
        #log_PxGh1 = -torch.sum(F.binary_cross_entropy(p.view(-1,784), inputs.view(-1,784), reduction = 'none'),-1)
        log_PxGh1 = -torch.sum(F.binary_cross_entropy(p.view(p.size()[0],p.size()[1],784), inputs.view(p.size()[0],p.size()[1],784), reduction = 'none'),-1)
        #log_PxGh1 = torch.sum(inputs*torch.log(p) + (1-inputs)*torch.log(1-p), -1)
        
        log_q_y = torch.log(q_y + 1e-20)
        g = Variable(torch.log(torch.Tensor([1.0/self.categorical_dim])).to('cuda'))
        g = g.double()
        #second_term = torch.sum(q_y*(g-log_q_y),dim=[2,3])
        second_term = torch.sum(z_unreshape*(g-log_q_y),dim=[2,3])
        log_weight = log_PxGh1 + second_term
        weight = torch.exp(log_weight)
        loss = -torch.mean(torch.log(torch.mean(weight, 0)))        
        return loss
