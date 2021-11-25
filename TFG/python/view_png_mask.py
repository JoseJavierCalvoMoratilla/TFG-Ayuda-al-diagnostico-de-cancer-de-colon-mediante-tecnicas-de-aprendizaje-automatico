# Alumno: Calvo Moratilla, José Javier
# Profesor: Gómez Adrian, Jon Ander
# UPV - ETSINF
# 10-02-2021
# Ayuda al diagnóstico de cáncer de colon mediante aprendizaje automático.

# Displays the medical image overlaid with the labeled mask.

# use python ./python/view_png_mask.py --folder-png ./data/paip2020/test/png_img_l3/ --folder-mask ./data/paip2020/test/mask_img_l3_predicted/ --number-img 2 --folder-out ./

# Import the libraries needed
from PIL import Image
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import sys, glob

# If don't have the 3 input arguments it raises an exception
if len(sys.argv) != 9:
    raise Exception( f' \n Unexpected number of inputs: \n \n Expected: \n --folder-png \n --folder-mask \n --number-image \n --folder-out \n')

# Obtain info by args
folders_input = []

for i in range(len(sys.argv)):
        if sys.argv[i] == '--folder-png':
           folders_input.append(str(sys.argv[i + 1]))
        elif sys.argv[i] == '--folder-mask':
            folders_input.append(str(sys.argv[i + 1]))
        elif sys.argv[i] == '--number-img':
           select = int(sys.argv[i + 1])
        elif sys.argv[i] == '--folder-out':
           folder_output = str(sys.argv[i + 1])

# The list of files in the respective folders is obtained.
list_img = sorted(glob.glob(folders_input[0] + '*.png'))
list_tif = sorted(glob.glob(folders_input[1] + '*.png'))

# Prepare the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,8), sharex = True, sharey = True)

# Loads the image and the selected mask
img = cv2.imread(list_img[select], cv2.IMREAD_UNCHANGED)
mask = cv2.imread(list_tif[select], cv2.IMREAD_UNCHANGED)

# Load the image in the plot
axes[0].imshow(img)

# Prepares the mask to be superimposed on the medical image
result = np.zeros(img.shape)
result[:,:,:] = img
result[:,:,1] += mask[:,:] * 0.5
result = np.minimum(result, 255).astype(np.uint8)

#Load the mask on the plot
axes[1].imshow(result)
plt.tight_layout()
#plt.show()

plt.savefig(folder_output + 'view_png_mask.png')
print('The plot has been created:')