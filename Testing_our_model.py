#!/usr/bin/env python
# coding: utf-8

# In[34]:


import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = nn.Sequential(nn.Linear(20, 64),
                      nn.ReLU(),
                      nn.Dropout(p=0.25),
                      nn.Linear(64, 256),
                      nn.ReLU(),
                      nn.Dropout(p=0.25),
                      nn.Linear(256, 256),
                      nn.ReLU(),
                      nn.Dropout(p=0.25),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(p=0.25),
                      nn.Linear(64, 2),
                      nn.Sigmoid()
                      ) 
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


# In[35]:


model = load_checkpoint('checkpoint.pth')
#model.state_dict


# In[38]:


final_test_data = pd.read_csv('output_audio_processing.csv')
final_test_data=final_test_data.iloc[:,3:] #as we want values from cloumn 2 to end ,can be checked from output_audio_processing.csv
final_test_data_tensor = torch.tensor(final_test_data.values) #converting to tensor 
print(final_test_data_tensor)
size_of_test_data=final_test_data_tensor.shape[0] #size to test_data
finaltestloader = torch.utils.data.DataLoader(final_test_data_tensor, batch_size=size_of_test_data, shuffle=False)

# In[39]:


with torch.no_grad():
    model.eval()
    for final_testing_data in finaltestloader:
        print(model(final_testing_data.type(torch.FloatTensor)))
        final_output_test_data = model(final_testing_data.type(torch.FloatTensor)).view(-1) #precdicted value we get from network
        final_output_test_data_numpy=final_output_test_data.numpy()
        print(final_output_test_data_numpy)
        if final_output_test_data_numpy[0] > final_output_test_data_numpy[1]:
          print("female")
        else:
          print("male")


# In[ ]:





# In[ ]:




