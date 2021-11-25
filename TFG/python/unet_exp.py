# Alumno: Calvo Moratilla, José Javier
# Profesor: Gómez Adrian, Jon Ander
# UPV - ETSINF
# 10-02-2021
# Ayuda al diagnóstico de cáncer de colon mediante aprendizaje automático.
# Programa para experimentar con los modelos unet entrenados 

#Import the libs
import argparse
import glob
import os
import re
import sys
from numpy.lib.arraypad import pad
import cv2
import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from tqdm import tqdm

#Search Patterns
buscarShapes = re.compile('[_][0-9]*[_][0-9]*[.]')

#Folder that cotains unet trained models
folder_input_models = '../models/'

#Folder that contains train images and mask
folder_input_png = '../data/paip2020/training/png_img_l3_mini/'
folder_input_mask = '../data/paip2020/training/mask_img_l3/'

#folder mask predicted
folder_predicted_mask = '../data/paip2020/training/mask_img_predicted/'
#if not exist folder, creates
os.makedirs(folder_predicted_mask, exist_ok=True)

#Obtain a list with absolute name of models files
#Obtain every name of data in input folder
list_models = sorted(glob.glob(folder_input_models + '*.onnx'))
list_png = sorted(glob.glob(folder_input_png + '*.png'))
list_mask = sorted(glob.glob(folder_input_mask + '*.tif'))

#number of images to compare in IoU
number_img_segment = len(list_mask) 

#Carga el modelo de un fichero ONNX
def cargarModelo(nombreModelo):
    #Available: LOG_LEVEL::{TRACE, DEBUG, INFO, WARN, ERROR, NO_LOGS}. (default LOG_LEVEL::INFO)
    return eddl.import_net_from_onnx_file(nombreModelo, input_shape=[4,176,176], mem=0, log_level=eddl.LOG_LEVEL.ERROR)

#Transforma una imagen de entrada (200, 200, 3) en un tensor de (3,176,176) realizando un crop central
def cargarImagenToTensor(nombreImagen, size, pad):
    imagen = (cv2.imread(nombreImagen) / 255.0).astype(np.float32)    
    (h, w) = size
    imagenCortada = imagen[pad:pad+h, pad:pad+w]   
    imagenPermutada = np.transpose(imagenCortada, axes = [2,0,1]) #.reshape(-1,3,176,176)         
    return Tensor.fromarray(imagenPermutada)

#Realiza la predicción del modelo
def predict(modelo, tensor):      
    return eddl.predict(modelo, tensor)


# FUNCIÓN joinImage: 
# Se encarga de unir las mask resultantes de la red unet
# INPUTS
# dirSalida --> Folder de dónde se guardan las imágenes unidas
# listaArxToJoin --> lista de archivos para hacer join
#   
def joinImages(listaArxToJoin, outShape):   
     
    #se crean el directorio de salida, si no existe
    #os.makedirs(dirOutput, exist_ok=True)

    #obtain output size
    h_ouput, w_ouput, ch = outShape

    #create the outputImage
    outputImage = np.zeros((h_ouput, w_ouput, ch), dtype=np.uint8)

    #Se recorren todos los archivos de esa carpeta y se guardan en img para ser recorridas y recortadas
    #Formato de entrada
    #nombreficheropadre_width_heigth.tif"

    for nombreImagenAbsoluto in tqdm(listaArxToJoin): 

        #se abre la imagen y se coloca un pad de 176x176 a 200x200
        #img = Ima ge.open(im) 
        
        #Buscar patron en el nombre
        shapes = buscarShapes.search(nombreImagenAbsoluto).group()
        
        #Se quitan las barras bajas para poder hacer split
        listShapes = re.sub("_", " ", shapes[0:-1]).split()      

           #se obtiene el desplazamiento horizontal y vertical de la imagen entrante
        origWidth = int(listShapes[1]) #col
        origHeight = int(listShapes[0]) #row  

        #EN CASO DE PAD  
        #img = np.pad(cv2.imread(nombreImagenAbsoluto),12)       
        img = cv2.imread(nombreImagenAbsoluto)
        
        #se obtienen el width, heigth correspondiente de la imagen
        height, width, _ = img.shape  

        #Se reconstruye la imagen grande
        outputImage[origHeight:origHeight+height, origWidth:origWidth+width] = np.maximum(outputImage[origHeight:origHeight+height, origWidth:origWidth+width], img[:,:])
    return outputImage

#calculate IoU score
def IoU(a, b):    
    intersection = np.logical_and(a, b)
    union = np.logical_or(a, b)
    return np.sum(intersection) / np.sum(union)


#Interate a list of models and do an experimentation to obtain the best model for use    
#for model in tqdm(list_models):  

#Config and load the unet
#Se carga el modelo
unet = cargarModelo(list_models[0])
#Hacemos el build de la red
eddl.build(unet, lo = None, me = None, cs = eddl.CS_GPU([1], mem = "low_mem")) # only in GPU 0 with low_mem setup

print('Iterating all segment of data to compute IoU for this Unet')
#iterate number total of images in segment
for i in tqdm(range(number_img_segment)):
    #obtain segment of images mini in interation
    list_png_iter = sorted(glob.glob(folder_input_png + 'training_data_' + str(i) + '*.png'))
    
    #predict every mini png in each iteration
    for name_png in list_png_iter:
        #read the image
        png = cv2.imread(name_png)
        #predict mask
        tensor_predicted_mask = predict(unet, [cargarImagenToTensor(png, (200,200), 12)])
        #transform tensor to NumPy array
        predicted_mask = Tensor.getdata(tensor_predicted_mask)
        #save the image with .tif extension
        cv2.imwrite(folder_predicted_mask + 'predict_' + name_png[-4] + '.tif' , predicted_mask) 
    
    #obtain the images was predicted and join 
    list_mask_pred_iter = sorted(glob.glob(folder_input_mask + 'predict_training_data_' + str(i) + '_' + '*.tif'))

    #obtain original mask in iteration
    mask_orig = cv2.imread(list_mask[i])
    #join the images and generate a predicted mask with the same shape that respective l3 mask
    mask_predict = Tensor.getdata(joinImages(list_mask_pred_iter, mask_orig.shape))

    #compute IoU
    print('The IoU for the model: ', (list_models[0], ' with mask number: ', i, ' is: ', IoU(mask_orig, mask_predict)))

print('Finish the compute of program')




