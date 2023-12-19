#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import torch
import numpy as np
import torch.nn as nn
import random
import torch.nn.functional as F
from matplotlib import pyplot as plt


# In[2]:


####################################################################################################################
Parser = argparse.ArgumentParser()
Parser.add_argument('--CheckPointPath', default='../LEGO')
Parser.add_argument('--NumEpochs', type=int, default=150)
Parser.add_argument('--Nc', type=int, default=32)
Parser.add_argument('--MiniBatchSize', type=int, default=4096, help='Size of the MiniBatch to use, Default:1')
Parser.add_argument('--Nn', type=int, default=2)
Parser.add_argument('--Nf',type=int, default=6)
Ddata = np.load(r'../LEGO/depth_maps_Lego_400.npz')
DNN_Depth = [Ddata[key] for key in Ddata]

Args, _ = Parser.parse_known_args()
CheckPointPath = Args.CheckPointPath
epochs = Args.NumEpochs
Nc = Args.Nc
batch_size = Args.MiniBatchSize
near_threshold = Args.Nn
far_threshold = Args.Nf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################################################################

### npz file already contains the data of images, poses , foacl length of the train images dataset.
def read_data(device):
    data = np.load("../LEGO/Images_LEGO_400.npz")  #home/ppaidi/MV Project/Images_LEGO_400.npz
    images = data["images"]
    im_shape = images.shape
    (num_images, H, W, _) = images.shape
    poses = data["poses"]
    poses = torch.from_numpy(poses).to(device)
    focal =  data["focal"]
    focal = torch.from_numpy(focal)
    images = torch.from_numpy(images).to(device)
    return images, poses, focal

#######################################################################################################################


def plot_figures(Epochs, log_loss):
    plt.figure(figsize=(10, 4))
    plt.plot(Epochs, log_loss)
    plt.title("loss")
    # plt.show()0
    plt.savefig("Results/Loss.png")

    #############################################

images, poses, focal = read_data(device)
height, width = images.shape[1:3]
N_encode = 6
lr = 5e-3
########################################################################################################################


# In[3]:


class tNerf(nn.Module):

    def __init__(self, filter_sz = 128, N_encode = 6):
        super(tNerf, self).__init__()

        # self.layer1 = nn.Linear(3+3*2*N_encode, filter_sz)
        # self.layer2 = nn.Linear(filter_sz, filter_sz)
        # self.layer3 = nn.Linear(filter_sz, 4)  
        
        self.layer1 = nn.Linear(3+3*2*N_encode, filter_sz)
        self.layer2 = nn.Linear(filter_sz, filter_sz)
        self.layer3 = nn.Linear(filter_sz, filter_sz)
        self.layer4 = nn.Linear(filter_sz, filter_sz)        
        self.layer5 = nn.Linear(filter_sz+3+3*2*N_encode, filter_sz)
        self.layer6 = nn.Linear(filter_sz, filter_sz)
        self.layer7 = nn.Linear(filter_sz, filter_sz)
        self.layer8 = nn.Linear(filter_sz, filter_sz)       
        self.layer9 = nn.Linear(filter_sz, 4) 
        
        
#         self.lin1 = nn.Sequential(nn.Linear(input_channels,width), nn.ReLU())
#         self.lin2 = nn.Sequential(nn.Linear(width, width), nn.ReLU())
#         self.lin3 = nn.Sequential(nn.Linear(width, width), nn.ReLU())
#         self.lin4 = nn.Sequential(nn.Linear(width, width), nn.ReLU())
        
#         self.lin5 = nn.Sequential(nn.Linear(width + input_channels, width), nn.ReLU())
#         self.lin6 = nn.Sequential(nn.Linear(width, width), nn.ReLU())
#         self.lin7 = nn.Sequential(nn.Linear(width, width), nn.ReLU())
#         self.lin8 = nn.Sequential(nn.Linear(width, width), nn.ReLU())
#         # self.volume_density = nn.Sequential(nn.Linear(width,1), nn.ReLU())

#         # self.lin10 = nn.Sequential(nn.Linear(width,width//2), nn.ReLU())
#         self.lin11 = nn.Sequential(nn.Linear(width,4))
        
                                    
    def forward(self, x):
        # x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        # x = self.layer3(x) 
        residual = x
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(torch.cat([x , residual], axis=-1)))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        x = self.layer9(x) 
        
#         residual = x
#         x = self.lin1(x)
#         x = self.lin2(x)
#         x = self.lin3(x)
#         x = self.lin4(x)
#         x = self.lin5(torch.cat([x , residual], axis=-1))
#         x = self.lin6(x)
#         x = self.lin7(x)
#         x = self.lin8(x)
#         # sigma = self.volume_density(x)
#         # x = self.lin10(x)
#         rgbs = self.lin11(x)

        return x


# In[4]:


def get_rays(h, w, f, pose, near, far, Nc, device):

    #making meshgrid of x and y shape 
    x = torch.linspace(0, w-1, w)
    y = torch.linspace(0, h-1, h)
    xi, yi = torch.meshgrid(x, y, indexing='xy')
    xi = xi.to(device)
    yi = yi.to(device)
    
    # normalized coordinates
    norm_x = (xi - w * 0.5) / f
    norm_y = (yi - h * 0.5) / f

    #direction unit vectors matrix   USING colmap and considering z direction as -1 and combinign with x and y 
    directions = torch.stack([norm_x, - norm_y, -torch.ones_like(xi)], dim = -1)
    directions = directions[..., None,:]   # this extra dimesnion is to suffice 4D array , with 3 points at 100, 100, 32 points.
    

    #camera matrix : 3x3 matrix from the 4x4 projection matrix from the pose matrix 
    rotation = pose[:3, :3]
    translation = pose[:3, -1]

    # converting the ray directions into the camera direction
    camera_directions = directions * rotation
    
    # changing the dimesnions into 100, 100, 3 and normalizing 
    ray_directions = torch.sum(camera_directions, dim = -1)
    ray_directions = ray_directions/torch.linalg.norm(ray_directions, dim = -1, keepdims = True)
    ray_origins =  torch.broadcast_to(translation, ray_directions.shape)

    #get the sample points HERE near and far are define NC th number of instances 
    depth_val = torch.linspace(near, far, Nc) 
    
    # Adding the noise to the depth values 
    noise_shape = list(ray_origins.shape[:-1]) + [Nc]
    noise = torch.rand(size = noise_shape) * (far - near)/Nc
    
    ## here the noise size is 100, 100, 32 and the depth_cal size is 32 , creating depth points at 100, 100, 32 
    # by adding noise and from the start of the rays  
    depth_val = depth_val + noise
    depth_val = depth_val.to(device)
    
    ## creating the query points of 100, 100, 32, 4 in the direction of camera pose by muliplying with the ray directions.    
    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_val[..., :, None]
    # print(query_points.shape)

    return ray_directions, ray_origins, depth_val, query_points


# In[5]:


def positional_encoding(x, L):
    gamma = [x]
    for i in range(L):
        gamma.append(torch.sin((2.0**i) * x))
        gamma.append(torch.cos((2.0**i) * x))
    gamma = torch.cat(gamma, axis = -1)
    return gamma


def mini_batches(inputs, batch_size): 
    return [inputs[i:i + batch_size] for i in range(0, inputs.shape[0], batch_size)]


# In[6]:


def render(radiance_field, ray_origins, depth_values):
    
    # radiance field is predictions of in RGB and Density 
    # ray origins are 100, 100, 3  with start points only
    # Depth values is ditance value sof rays at 100, 100, 32
    
    #volume density
    sigma_a = F.relu(radiance_field[...,3])    # output of this is 100, 100, 32      
    
    #color value at nth depth value
    rgb = torch.sigmoid(radiance_field[...,:3])      # output of tis 100, 100, 32, 3
    
    # applying the formuale for the volume rendering
    one_e_10 = torch.tensor([1e10], dtype = ray_origins.dtype, device = ray_origins.device)
    
    dists = torch.cat((depth_values[...,1:] - depth_values[...,:-1], one_e_10.expand(depth_values[...,:1].shape)), dim = -1)
    
    alpha = 1. - torch.exp(-sigma_a * dists) 
    
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)     #transmittance
    
    # getting the rgb map
    rgb_map = (weights[..., None] * rgb).sum(dim = -2)          #resultant rgb color of n depth values
    
    # getting the depth map
    depth_map = (weights * depth_values).sum(dim = -1)
    
    # getting the accuracy map 
    acc_map = weights.sum(-1)
    
    return rgb_map, depth_map, acc_map

# To calculate the cumulative product used to calculate alpha
def cumprod_exclusive(tensor) :
    dim = -1
    cumprod = torch.cumprod(tensor, dim)
    cumprod = torch.roll(cumprod, 1, dim)
    cumprod[..., 0] = 1.

    return cumprod


# In[7]:


def training(h, w, f, pose, near, far, Nc, batch_size, N_encode, model, device):
    
    ## getting query points of 100, 100, 32, 3
    ray_directions, ray_origins, depth_values, query_points = get_rays(h, w, f, pose, near, far, Nc, device)
    
    # flattening shape of 100, 100, 32 , 3 into 320000, 3
    flat_query_pts = query_points.reshape((-1,3))
      
    #positional encoding combining 320000, 3 into sin and cosine waves and collecting agian, resulting 320000, 39
    encoded_points = positional_encoding(flat_query_pts, N_encode)
    
    # taking in minibatches for the training into model
    batches = mini_batches(encoded_points, batch_size = batch_size)
    
    ## every batch is of 4096, 39
    predictions = []

    ## training in batch 
    for batch in batches:  
        predictions.append((model(batch)))
     
    ## output of the prediction and stacking them to make of original size and 320000, 4 : 4 is output size of model
    radiance_field_flat = torch.cat(predictions, dim=0)

    # creating a list of elements 100, 100, 32 , 4 , not required this line , just to reshape the above prediction tensor 
    # into 100, 100, 32, 4 from 320000, 4 by creating a dummy 
    unflat_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field_flat, unflat_shape)
    
    ### columetric rendering 
    logits_rgb, _, D = render(radiance_field, ray_directions, depth_values)
    
    return logits_rgb, D


# In[8]:


def TrainOperation(images, poses, focal, height, width, lr, N_encode, epochs,
                   near_threshold, far_threshold, batch_size, Nc, device,DNN_Depth):

    model = tNerf()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    Loss = []
    Depth_Loss =[]
    Epochs = []
    Fin=[]
    show_iter = 50
    
    for i in range(epochs):
        print(i)

        img_idx = random.randint(0, images.shape[0]-7)

        
        from matplotlib import pyplot as plt

        img = images[img_idx].to(device)
        pose = poses[img_idx].to(device)

        rgb_logit,D = training(height, width, focal, pose, near_threshold, far_threshold, Nc, batch_size, N_encode, model, device)
        loss = F.mse_loss(rgb_logit, img) #photometric loss
            
        Loss2= F.mse_loss(torch.tensor(DNN_Depth[img_idx]), D)
        Loss2 = loss + (Loss2)
      
        
        if i % show_iter == 0:
            print("Loss", loss.item())
            Loss.append(loss.item())
            # Depth_Loss.append(Loss2.item())
            Epochs.append(i+1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        SaveName =  '../LEGO/model.ckpt'
        Fin.append((Loss, Loss2))
        # torch.save({'epoch': Epochs,
        #   'model_state_dict': model.state_dict(),
        #   'optimizer_state_dict': optimizer.state_dict()},
        #     SaveName)
        
    print(Epochs)
    
    plt.figure(figsize=(10, 4))
    plt.plot(Epochs, Loss)
    print(Loss)
    # print(Depth_Loss)
    # plt.plot(Epochs, Depth_Loss)

    plot_figures(Epochs, Loss)
    
    

    SaveName =  '../LEGO/model1.ckpt'

    torch.save({'epoch': Epochs,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict()},
            SaveName)  
    return Fin


# In[ ]:


Fin =TrainOperation(images, poses, focal, height, width, lr, N_encode, epochs, near_threshold, far_threshold, batch_size, Nc, device,DNN_Depth)


# In[ ]:





# In[ ]:




