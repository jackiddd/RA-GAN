# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 22:45:32 2021

@author: win10
"""
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import collections
from sklearn import datasets
import time

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        img, target = self.images[index], self.labels[index]
        img = torch.FloatTensor(img)
        target = torch.FloatTensor(target)
        return img, target

    def __len__(self):
        return len(self.images)
'''============ Select hyperparameters ==================='''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(47)

z_dim=16
lr=0.0001
batchsize=8
b1=0.5
b2=0.9
n_epochs=12001
n_critic=5
num=5000
lambda_gp=0.5

num_train = 100
'''============ Import data ==================='''
    
dataset= 'CO2'  #'diabete' 
method='RAGAN/' 
 

if dataset == 'CO2':
    
    data_dim=12 

    x= np.load('data'+'/'+dataset+'/x.npy')
    y= np.load('data'+'/'+dataset+'/y.npy').reshape(-1,1)
    
    #打乱数据
    array = np.arange(0, len(x))
    np.random.seed(2)
    np.random.shuffle(array)
    x_less=x[array]
    y_less=y[array]
    
    data_X = np.hstack((x_less,y_less))[:num_train,:]
    
    #封装数据
    data=MyDataset(data_X,np.ones([num_train,1]))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batchsize, shuffle=True, drop_last=False)

    data_all=torch.tensor(data_X).float()

if dataset == 'diabete':
    
    data_dim=11  

    data=datasets.load_diabetes()
    data_X=data.data
    data_Y=data.target
    
    #打乱数据
    array = np.arange(0, len(data_X))
    np.random.seed(2)
    np.random.shuffle(array)    
    x_less=data_X[array]
    y_less=data_Y[array]
    
    data_all = np.hstack((x_less,y_less.reshape(-1,1)))
    #标准化
    data_all=(data_all-data_all.min(0))/(data_all.max(0)-data_all.min(0))
    
    #封装数据
    data=MyDataset(data_all[:num_train,:],np.ones([num_train,1]))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batchsize, shuffle=True, drop_last=False)

    data_all=torch.tensor(data_all[:num_train,:]).float()

'''============ Define model structure ==================='''

class DNN(nn.Module):
    def __init__(self):
        super(DNN,self).__init__()
        self.mlp= nn.Sequential(
                nn.Linear(data_dim-1,32),
                nn.ReLU(),
                nn.Linear(32,8),
                nn.ReLU(),
                nn.Linear(8,1)
                )
        
    def forward(self, x):
        y = self.mlp(x)
        return y

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.G=nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU( inplace=True),
            
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU( inplace=True),
            
            nn.Linear(32,data_dim),                            
            nn.Sigmoid()
            )
        
        self.mlp=nn.Sequential(
            nn.Linear(data_dim-1,32),
            nn.LeakyReLU(),
            nn.Linear(32,8),
            nn.LeakyReLU(),
            nn.Linear(8,1),
            )
    
    def forward(self, z,z_x):
        x = self.G(z)
        y_1 = self.mlp(x[:,:-1])  
        y_2 = self.mlp(z_x[:,:-1])  
        
        return x,x[:,-1],y_1, y_2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.mlp = nn.Sequential(            
            nn.Linear(data_dim-1 ,32 ),
            nn.LeakyReLU( inplace=True),            
            nn.Linear(32 ,8 ),
            nn.LeakyReLU(inplace=True),
            )
        
        self.fc = nn.Linear(8,1)
        
        self.D = nn.Sequential(
            nn.Linear(8+data_dim,32 ),            
            nn.LeakyReLU( inplace=True),       
            nn.Linear(32 ,32 ),            
            nn.LeakyReLU( inplace=True),            
            nn.Linear(32, 1),            
            )
        
    def forward(self, img):
        fea = self.mlp(img[:,:-1])
        y = self.fc(fea)
        x = self.D(torch.cat((fea,img),1))
        
        return x, img[:,-1], y 
    
model= DNN()
generator = Generator()
discriminator = Discriminator()

'''=========== Pre-train DNN =============='''
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
    generator.cuda()
    discriminator.cuda()
    
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer_DNN = torch.optim.Adam(model.parameters(), lr=0.0001)
def train_DNN(epoch):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):        
        real_imgs = inputs[:,:-1]
        y=model(real_imgs)
        loss_y=loss_fn(y,inputs[:,-1].reshape(-1,1))
        optimizer_DNN.zero_grad()
        loss_y.backward()
        optimizer_DNN.step()

for epoch in range(401):
    loss=train_DNN(epoch)

torch.save(model.state_dict(), 'data'+'/'+dataset+'_model_DNN') #Backups
print('Finish training of DNN')

'''=============== Train RA-GAN and generate data ===================='''
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.shape[0], 1))   )
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates,_,_ = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

#Loading parameters
G_dict_trained =torch.load('data'+'/'+dataset+'_model_DNN')
G_dict_new =  generator.state_dict()
G_state_dict = {k:v for k,v in G_dict_trained.items() if k in G_dict_new.keys()}
G_dict_new.update(G_state_dict)
generator.load_state_dict(G_dict_new)
for p in generator.mlp.parameters():
    p.requires_grad = False

D_dict_trained =torch.load('data'+'/'+dataset+'_model_DNN')
D_dict_new =  discriminator.state_dict()
D_state_dict = {k:v for k,v in D_dict_trained.items() if k in D_dict_new.keys()}
D_dict_new.update(D_state_dict)
discriminator.load_state_dict(D_dict_new)

#Training
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1,b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer_D.state = collections.defaultdict(dict)
optimizer_G.state = collections.defaultdict(dict)

t3= time.time()

w_distance_all= []
G_RAL_all = []
for epoch in range(n_epochs):
    WD=0
    GRAL=0
    for i, (imgs, _) in enumerate(dataloader):
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        optimizer_D.zero_grad()
        z = Variable(Tensor(np.random.uniform(0, 1, (imgs.shape[0], z_dim))))

        # Generate a batch of images
        fake_imgs, _, _ ,_= generator(z,data_all.cuda())

        real_validity, y_d_real, d_y_real = discriminator(real_imgs)
        fake_validity, y_d_fake, d_y_fake = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        w_distance = torch.mean(real_validity) - torch.mean(fake_validity)
        
        WD +=  w_distance
        #RA_loss
        KAL_fake= 1 *loss_fn(y_d_fake.reshape(-1,1),d_y_fake.reshape(-1,1))   
        KAL_real= 5 *loss_fn(y_d_real.reshape(-1,1),d_y_real.reshape(-1,1))
        
        d_loss_all= d_loss  + KAL_real
        
        d_loss_all.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        
        if i % n_critic == 0:

            fake_imgs,g_y, y_g,ygg = generator(z,data_all.cuda()) 
            G_RAL = 5*loss_fn(g_y.reshape(-1,1),y_g.reshape(-1,1))  
            # G_add = 5*loss_fn(ygg.reshape(-1,1),data_all.cuda()[:,-1].reshape(-1,1)) 
            
            GRAL += G_RAL

            fake_validity,_,_ = discriminator(fake_imgs)
            g_loss = - torch.mean(fake_validity) + G_RAL

            g_loss.backward()
            optimizer_G.step()
    
    w_distance_all.append(WD/i)
    G_RAL_all.append(GRAL/i)
    
    if epoch % 1000 == 0:
    
        generator.eval()
    
        z = Variable(Tensor(np.random.uniform(0, 1, (num, z_dim)))) 
        data,_,_,_= generator(z,data_all.cuda())
    
        np.save('data/'+method+dataset+'_data_%d.npy'%epoch, data.cpu().detach().numpy())        
    
    print(epoch)
    
print('Finish training of RAGAN')

w_distance_all=torch.tensor(w_distance_all)
G_RAL_all=torch.tensor(G_RAL_all)

torch.save(w_distance_all,'data/'+method+dataset+'_wd.npy') 
torch.save(G_RAL_all,'data/'+method+dataset+'_GRAL.npy')  

torch.cuda.synchronize()
t4 = time.time()
print('Time is {}'.format(t4-t3))







