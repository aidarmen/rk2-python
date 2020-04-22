import numpy as np 
import pandas as pd 
import ast
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps

def view_image(img, width = 256, height = 256):
    fig, ax = plt.subplots(figsize=(6,9))
    ax.imshow(img.reshape(width, height).squeeze())
    ax.axis('off')

    plt.show()

def view_images_grid(X, y):

    fig, axs = plt.subplots(5, 5, figsize=(50,50))
    
    for label_num in range(0,25):
        r_label = random.randint(0, len(X) - 1)
        image = X[r_label].reshape(28,28)  #reshape images
        i = label_num // 5
        j = label_num % 5
        axs[i,j].imshow(image) #plot the data
        axs[i,j].axis('off')
        
        axs[i,j].set_title( label_dict[y[r_label]] ,fontsize=50)

    plt.show()