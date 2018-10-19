import numpy as np
import torch
import torch.utils.data as data
from glob import glob
import os
import os.path
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
import torchvision.utils as v_utils
from utils import *

class Bee_Dataset(data.Dataset):

    def __init__(self, img_dir,csv_dir):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_dir)
        #print(self.df.loc[:,'file'])
        print(self.df.subspecies.unique())
        # >>> df.health.unique()
        # array(['hive being robbed', 'healthy', 'few varrao, hive beetles',
        #     'ant problems', 'missing queen', 'Varroa, Small Hive Beetles'],
        #     dtype=object)


    def __getitem__(self, index):

        filename = self.df.loc[index,'file']
        
        #print(self.df.loc[index,'subspecies'])
        if self.df.loc[index,'subspecies'] == '-1':
            subspecies = 0
        elif self.df.loc[index,'subspecies'] == 'Italian honey bee':
            subspecies = 1
        elif self.df.loc[index,'subspecies'] == 'VSH Italian honey bee':
            subspecies = 2
        elif self.df.loc[index,'subspecies'] == 'Carniolan honey bee':
            subspecies = 3
        elif self.df.loc[index,'subspecies'] == 'Russian honey bee':
            subspecies = 4
        elif self.df.loc[index,'subspecies'] == '1 Mixed local stock 2':
            subspecies = 5
        elif self.df.loc[index,'subspecies'] == 'Western honey bee':
            subspecies = 6

        if self.df.loc[index,'health'] == 'healthy':
            health = 0
        elif self.df.loc[index,'health'] == 'hive being robbed': 
            health = 1
        elif self.df.loc[index,'health'] == 'few varrao, hive beetles':
            health = 2  
        elif self.df.loc[index,'health'] == 'ant problems' :
            health = 3     
        elif self.df.loc[index,'health'] == 'missing queen':
            health = 4
        elif self.df.loc[index,'health'] == 'Varroa, Small Hive Beetles': 
            health = 5

        

        img = load_img(os.path.join(self.img_dir,filename))

        sample = {'image':img, 'subspecies':subspecies,'health':health}
        #Some pytorch loss function doesn't need one-hot

        return sample

    def __len__(self):
        return len(self.df)