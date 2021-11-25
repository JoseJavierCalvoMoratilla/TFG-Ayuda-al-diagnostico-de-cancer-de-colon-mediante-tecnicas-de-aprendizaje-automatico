# Alumno: Calvo Moratilla, José Javier
# Profesor: Gómez Adrian, Jon Ander
# UPV - ETSINF
# 29-04-2021
# Ayuda al diagnóstico de cáncer de colon mediante aprendizaje automático.

# Code provided by Jon Ander Gómez Adrian

# Image shuttle to neural network

# Import the libraries needed
import sys
import numpy as np
import pandas as pd
#from tensorflow import keras
#from PIL import Image
import cv2
import os, glob, re
from tqdm import tqdm
from random import shuffle


class DataGenerator: # (keras.utils.Sequence):
    'Generates batches of samples for Keras'

    def __init__(self,  folder_png = './png_img_l3_mini/',
                        folder_mask = './mask_img_l3_mini/',
                        batch_size = 32,
                        shuffle = True,
                        verbose = False,
                        for_softmax = True,
                        use_cache = True,
                        return_filenames = False):
        '''
            Initialization
     

        '''
        #Variables para la clase
        self.verbose = verbose #Verbose
        self.folder_png = folder_png #Folder pngs
        self.folder_mask = folder_mask #Folders mask
        self.batch_size = batch_size #Tamany del batch 32
        self.shuffle = shuffle #Mezclar
        self.dict_png = []
        self.dict_mask = []
        self.index = []         
        self.for_softmax = for_softmax
        self.use_cache = use_cache
        self.return_filenames = return_filenames

        #Obtain the list of input data      
        self.filenames = [s[len(self.folder_png):] for s in glob.glob(self.folder_png + '/' + '*.png')]
        self.filenames = [s[: -4] for s in self.filenames] 

        #lista ordenada de 0 a numImg
        self.index = np.arange(len(self.filenames))

        if self.use_cache:
            self.prefetch()

        self.on_epoch_end()
        print('Data generator ready with %d samples' % len(self.index))

    def prefetch(self):
        self.cache = list()
        for filename in self.filenames:
            print('prefetching ', filename, flush = True)
            filename_png  = self.folder_png  + '/' + filename + '.png'
            filename_mask = self.folder_mask + '/' + filename + '.png' # '.tif'

            image_png  = cv2.imread(filename_png,  cv2.IMREAD_UNCHANGED)
            image_mask = cv2.imread(filename_mask, cv2.IMREAD_UNCHANGED)

            image_png = image_png / 255.
            image_mask[image_mask > 0] = 1
            image_mask[image_mask < 1] = 0
            image_mask = image_mask.astype(int)

            if self.for_softmax:
                image_mask = np.vstack([[1 - image_mask], [image_mask]])
            else:
                image_mask = np.expand_dims(image_mask, axis = 0)
            '''
            image_mask = np.vstack([[1 - image_mask], [image_mask]])
            '''
            #
            n_channels = image_png.shape[2] 
            # we are not sure if channel 4 is alpha, so we work with the four channels
            image_png = np.array([image_png[:,:,i] for i in range(n_channels)])
            #
            self.cache.append((image_png, image_mask))
    # ------------------------------------------------------------------------------------

    def __len__( self ):
        'Denotes the number of batches per epoch'
        if len(self.index) % self.batch_size == 0:
            return len(self.index) // self.batch_size
        else:
            return len(self.index) // self.batch_size + 1    

    def on_epoch_end( self ):
        'Updates indexes after each epoch'
        if self.shuffle:
            shuffle(self.index)       

    #Nº de batch
    def __getitem__(self, batch_index):
        'Generates one batch of data'
        #
        X = []
        Y = []
        F = []

        if self.use_cache:

            for i in range(self.batch_size):
                j = (batch_index * self.batch_size + i) % len(self.index)
                k = self.index[j] 

                X.append(self.cache[k][0])
                Y.append(self.cache[k][1])             

        else:

            for i in range(self.batch_size):
                #Pos dintre del index
                j = (batch_index * self.batch_size + i) % len(self.index)
                #Pos dintre de filenames
                k = self.index[j] 
                filename_png = self.folder_png + '/' + self.filenames[k] + '.png'
                filename_mask = self.folder_mask + '/' + self.filenames[k] + '.png' # '.tif'

                image_png = cv2.imread(filename_png, cv2.IMREAD_UNCHANGED)
                image_mask = cv2.imread(filename_mask, cv2.IMREAD_UNCHANGED)


                image_png = image_png / 255.
                #image_png = 1. - image_png / 255. # just a test, no improvement observed
                #image_mask = image_mask / 255.
                image_mask[image_mask > 0] = 1
                image_mask[image_mask < 1] = 0
                image_mask = image_mask.astype(int)

                if self.for_softmax:
                    image_mask = np.vstack([[1 - image_mask], [image_mask]])
                else:
                    image_mask = np.expand_dims(image_mask, axis = 0)
                '''
                image_mask = np.vstack([[1 - image_mask], [image_mask]])
                '''

                #print('image', image_png.min(), image_png.max(), image_png.ptp(), 'mask', image_mask.min(), image_mask.max(), image_mask.ptp())

                n_channels = image_png.shape[2] 
                image_png = np.array([image_png[:,:,i] for i in range(n_channels)]) # we are not sure if channel 4 is alpha, so we work with the four channels

                #print("PNG", image_png.shape, "TIF", image_mask.shape)
                
                X.append(image_png)
                Y.append(image_mask)             
                F.append(self.filenames[k])
        # end of else
       
        X = np.array(X)
        Y = np.array(Y)
 
        #
        if self.return_filenames:
            return X, Y, F
        else:
            return X, Y
    

         


if __name__ == '__main__':
    base_data_dir = './data/paip2020/training'
    dg = DataGenerator(folder_png= f'{base_data_dir}/png_img_l3_mini/', folder_mask = f'{base_data_dir}/mask_img_l3_mini/' , batch_size=32, shuffle=True, verbose=False)
    for i in range(len(dg)):
        x,y, = dg[i] # .__getitem__(i)
        print(i, x.shape, y.shape,
                 [x[:,i,:,:].min() for i in range(4)],
                 [x[:,i,:,:].mean() for i in range(4)],
                 [x[:,i,:,:].max() for i in range(4)],
                 y[:,:,:].min(), y[:,:,:].max(), y[:,:,:].sum())
