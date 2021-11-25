# Alumno: Calvo Moratilla, José Javier
# Profesor: Gómez Adrian, Jon Ander
# UPV - ETSINF
# 10-02-2021
# Ayuda al diagnóstico de cáncer de colon mediante aprendizaje automático.

# Checks if the size of the images in a folder corresponds to the size of their respective masks

# use python ./python/inspect_data.py --folder-png data/paip2020/training/png_img_l3/ --folder-mask data/paip2020/training/mask_img_l3/

# Import the libraries needed
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os, glob, re
import sys


# If don't have the 2 input arguments it raises an exception
if len(sys.argv) != 5:
    raise Exception( f' \n Unexpected number of inputs: \n \n Expected: \n --folder-png \n --folder-mask \n  ')

# Obtain info by args
folders_input = []

for i in range(len(sys.argv)):
        if sys.argv[i] == '--folder-png':
           folders_input.append(str(sys.argv[i + 1]))
        elif sys.argv[i] == '--folder-mask':
            folders_input.append(str(sys.argv[i + 1]))


# The list of files in the respective folders is obtained.
list_img = sorted(glob.glob(folders_input[0] + '*.png'))
list_tif = sorted(glob.glob(folders_input[1] + '*.tif'))

# Run the process
if __name__ == '__main__':   
    for filename_1, filename_2 in zip(list_img, list_tif):
        png = Image.open(filename_1)
        tif = Image.open(filename_2)
        print(png.size, tif.size, filename_1, filename_2, flush = True)
        if png.size != tif.size:
            raise Exception('different sizes between image and mask')
        png.close()
        tif.close()