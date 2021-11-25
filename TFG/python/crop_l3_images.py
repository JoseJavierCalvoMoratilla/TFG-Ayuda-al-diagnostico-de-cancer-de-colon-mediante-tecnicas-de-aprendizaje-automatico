# Alumno: Calvo Moratilla, José Javier
# Profesor: Gómez Adrian, Jon Ander
# UPV - ETSINF
# 10-02-2021
# Ayuda al diagnóstico de cáncer de colon mediante aprendizaje automático.

# The program cut medical images and their respective mask with a little size to prepare the input of the model.

# use python ./python/crop_l3_images.py --folder-png data/paip2020/training/png_img_l3/ --folder-mask data/paip2020/training/mask_img_l3/

# Import the libraries needed
import numpy as np
from tqdm import tqdm
import os, glob, re
import cv2
import sys

# Vars to cutting images task
# Increment height and width in cut task 64 by 64 pixels
incX, incY = (64, 64)
#Size of cut image
w, h = (128, 128)

# If don't have the 2 input arguments it raises an exception
if len(sys.argv) != 5:
    raise Exception( f' \n Unexpected number of inputs: \n \n Expected: \n --folder-png \n --folder-mask \n  ')

#Obtain info by args
folders_input = []

for i in range(len(sys.argv)):
        if sys.argv[i] == '--folder_png':
           folder_input.append(str(sys.argv[i + 1]))
        elif sys.argv[i] == '--folder_mask':
            folder_input.append(str(sys.argv[i + 1]))
       

# Obtains the list of files inside the folders 
list_img = sorted(glob.glob(folders_input[0] + '*.png'))
list_tif = sorted(glob.glob(folders_input[1] + '*.tif'))

# Var to save the name of the original image used
padreImg = ''

# Function cropImage: 
# Cuts image with a specific characteristics
# INPUTS
# input --> Folder name of input data
# listaArx --> List with the relative paths to obtain images
# extension --> Extension of output file          
def cropImage(entrada, listaArx, extension):

    # Creates the output folder if it doesn't exist
    dest_dir = entrada[:-1] if entrada.endswith('/') else entrada
    dest_dir += f'.{h}x{w}'
    os.makedirs(dest_dir, exist_ok = True)

    # Iterate the whole list of input images images to cut them   
    for im in listaArx:

        # Save the name of the father/ mother original image
        padreIni = im.rfind('/')
        lenExt = len(extension)
        padreImg = im[padreIni + 1: -lenExt]

        # Read image by OpenCV Library
        img = cv2.imread(im, cv2.IMREAD_UNCHANGED)  

        # Depend on the channels of image height, width, channels vars are defined                             
        if len(img.shape) == 2:
            height, width = img.shape
            ch = 1
        elif len(img.shape) == 3:
            height, width, ch = img.shape
        else:
            raise Exception(f'Unexpected shape for file {im}')

        print(im, img.shape, img.min(), img.max(), img.ptp())

        # Iterates every image to crop a little relative segment of the original image 
        #with sizes as 128 by 128 pixels with a window-sized as 64 by 64 pixels.
        for row_i in range(0, height, incY):
            for col_i in range(0, width, incX):          
                _h = min(h, height - row_i)
                _w = min(w, width - col_i)
                if ch > 1:
                    crop = np.zeros([h, w, ch])
                    crop[0: _h, 0: _w, :] = img[row_i : row_i + _h, col_i : col_i + _w, :]
                else:
                    crop = np.zeros([h, w])
                    crop[0: _h, 0: _w] = img[row_i : row_i + _h, col_i : col_i + _w]
                
                save_to = os.path.join(dest_dir, padreImg + "_" + str(row_i) + "_" + str(col_i)  + extension) #"data_{:0000000000000009}"
                
                # Save the cropped image in an output folder with a replacement of string segment  
                #'annotation tumor'
                cv2.imwrite(save_to.replace('annotation_tumor_', ''), crop)
              
# Run the process
if __name__ == '__main__':    
    print()
    print('The process starts with the png images... \n')     
    cropImage(folders_input[0], list_img, '.png')
    print('The process with the png images is finished.\n')  
    print('The process starts with the tif masks ... \n')     
    cropImage(folders_input[1], list_tif, '.png')
    print('The process with the masks is finished. tif \n')  
    print("The dataset creation process is complete.") 
