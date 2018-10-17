from PIL import Image
import cv2

import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import matplotlib.cm as cm
import torch

def load_img(img_dir,resize_shape=[224,224]):
    '''
    Load and resize image

    Args:
        img_dir
    Returns: 
        img: numpy array with pytorch dimension
    '''
    #img = Image.open(img_dir)
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224, 224))

    #img = np.array(img)
    #print(img.shape)
    img = np.transpose(img,(2,0,1))# Swap axis
    #img = img.resize(resize_shape,Image.BICUBIC)

    #Normalize between 0 and 1
    img = img*(1.0/img.max())
    
    return img


def visualize_predictions(model,img,label,num_images=8):

    images_so_far = 0
    fig = plt.figure()
    
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        for j in range(num_images):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}, ground truth{}'.format(preds[j],label[j]))
            to_show = img.cpu().data[j]
            #print(to_show.shape)
            to_show = np.transpose(to_show,[1,2,0])
            plt.imshow(to_show)

    plt.savefig('predictions')
    plt.close('all')






