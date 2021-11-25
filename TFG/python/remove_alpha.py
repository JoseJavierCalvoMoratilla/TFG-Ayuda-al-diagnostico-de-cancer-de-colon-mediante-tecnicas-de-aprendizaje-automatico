# Alumno: Calvo Moratilla, José Javier
# Profesor: Gómez Adrian, Jon Ander
# UPV - ETSINF
# 29-04-2021
# Ayuda al diagnóstico de cáncer de colon mediante aprendizaje automático.

# Removes the alpha channel from images

# use python ./python/remove_alpha.py --folder-input ./data/paip2020/training/png_img_l3/ --folder-output ./data/paip2020/training/png_img_l3_noalpha

# Import the libraries needed
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os, glob, re, sys

# If don't have the 2 input arguments it raises an exception
if len(sys.argv) != 5:
    raise Exception( f' \n Unexpected number of inputs: \n \n Expected: \n --folder-input \n --folder-output \n  ')

# Obtain info by args
folders_input = []

for i in range(len(sys.argv)):
        if sys.argv[i] == '--folder-input':
           folder_input = str(sys.argv[i + 1])
        elif sys.argv[i] == '--folder-output':
            folder_output = str(sys.argv[i + 1])

# Gets a list of png images from folder 
list_img = sorted(glob.glob(folder_input + '*.png'))

# Function removeAlpha: 
# Deletes the alpha channel
# INPUTS
# input --> Image to remove the alpha channel
def removeAlpha(img):
    # In case of grayScale images the len(img.shape) == 2
    if len(img.shape) > 2 and img.shape[2] == 4:        
        array_noalpha = img[:,:,:3]
        image_noalpha = Image.fromarray(array_noalpha, 'RGB')        
    return image_noalpha

# Create the output folder if it does not exist
os.makedirs(folder_output, exist_ok=True)

# Iterate over the images in the input folder and delete the alpha channel.
if __name__ == '__main__': 
    counter = 1
    print('Starts transform proces, RGB Alpha to RGB')
    for image in tqdm(list_img):
        read_png = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        transformed_png = removeAlpha(read_png)
        transformed_png.save(folder_output + "data_noalpha_" + str(counter) + ".png")
        counter += 1
    print('Transform process ends')







