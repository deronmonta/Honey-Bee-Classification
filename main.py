from model import *
from dataset import *
from utils import *

from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_dir', default='./data/bee_imgs', help='path to directory containing the images')
parser.add_argument('--csv_dir', default='./data/bee_data.csv', help='path to directory containing the csv file')

parser.add_argument('--num_filters',type=int, default=8, help='')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate for both model')
opts = parser.parse_args()
print(opts)

NUM_CLASS = 7


bee_dataset = Bee_Dataset(img_dir=opts.img_dir,csv_dir=opts.csv_dir)
dataloader = DataLoader(bee_dataset,batch_size=opts.batch_size,shuffle=True,drop_last=True,num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet152(pretrained=True)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASS)
#print(model_ft)


try:
    model_ft = torch.load('./model/resnet_finetune.pkl')
    print("\n----------------------Model restored----------------------\n")
except:
    print("\n----------------------Model not restored----------------------\n")
    pass

model_ft = model_ft.to(device)  



criterion =  nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=opts.learning_rate,betas=(0.9, 0.999),amsgrad=True)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

for epoch in tqdm(range(opts.epochs)):
    for i, sample in tqdm(enumerate(dataloader)):
        #print(sample)
        img = sample['image'].float().cuda()
        subspecies = sample['subspecies'].long().cuda()

        prediction = model_ft.forward(img)
        loss = criterion(prediction,subspecies)
        loss.backward()
        optimizer_ft.step()

        if i % 100 == 0:
            print('Loss: {}'.format(loss))
            visualize_predictions(model_ft,img,subspecies)

            torch.save(model_ft,'./model/resnet_finetune.pkl')
            
            
            

    