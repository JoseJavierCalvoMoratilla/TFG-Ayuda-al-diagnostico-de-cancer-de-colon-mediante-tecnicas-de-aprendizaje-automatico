# Alumno: Calvo Moratilla, José Javier
# Profesor: Gómez Adrian, Jon Ander
# UPV - ETSINF
# 10-02-2021
# Ayuda al diagnóstico de cáncer de colon mediante aprendizaje automático.

# Program for the definition of the U-Net network architecture to be used, training and subsequent evaluation of the network.

# Import the libraries needed
import argparse
import sys
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import numpy as np
import cv2
from dataGenerate import DataGenerator

USE_CONCAT = 1

MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")



# Function for choosing a U-Net architecture variant 1 a or b
#Input: Layer x
def UNetWithPadding_1(x, with_sigmoid = False, variant = 'a'):
    depth=32  
    x1 = x
    #
    x1 = eddl.Conv(x1, 1 * depth, [3, 3], [1, 1], "same")  # 128 x 128
    x1 = eddl.Conv(x1, 1 * depth, [3, 3], [1, 1], "same")
    x1 = eddl.ReLu(x1)
    #
    x2 = eddl.MaxPool(x1, [2, 2], [2, 2])
    #
    x2 = eddl.Conv(x2, 2 * depth, [3, 3], [1, 1], "same")  # 64 x 64
    x2 = eddl.Conv(x2, 2 * depth, [3, 3], [1, 1], "same")
    x2 = eddl.ReLu(x2)
    #
    x3 = eddl.MaxPool(x2, [2, 2], [2, 2])
    #
    x3 = eddl.Conv(x3, 3 * depth, [3, 3], [1, 1], "same") # 32 x 32
    x3 = eddl.Conv(x3, 3 * depth, [3, 3], [1, 1], "same")
    x3 = eddl.ReLu(x3)
    #
    x4 = eddl.MaxPool(x3, [2, 2], [2, 2])
    #
    x4 = eddl.Conv(x4, 4 * depth, [3, 3], [1, 1], "same") # 16 x 16
    x4 = eddl.Conv(x4, 4 * depth, [3, 3], [1, 1], "same")
    x4 = eddl.ReLu(x4)
    #
    x5 = eddl.MaxPool(x4, [2, 2], [2, 2])
    #
    x5 = eddl.Conv(x5, 5 * depth, [3, 3], [1, 1], "same") # 8 x 8
    x5 = eddl.Conv(x5, 5 * depth, [3, 3], [1, 1], "same")
    x5 = eddl.ReLu(x5)
    #
    x6 = eddl.MaxPool(x5, [2, 2], [2, 2])
    #
    x6 = eddl.Conv(x6, 6 * depth, [3, 3], [1, 1], "same") # 4 x 4
    x6 = eddl.Conv(x6, 6 * depth, [3, 3], [1, 1], "same")
    x6 = eddl.ReLu(x6)
    #
    #
    neck = eddl.Flatten(x6)
    #
    l = neck
    l = eddl.Dense(l, 6 * depth * 4 * 4)
    l = eddl.ReLu(l)
    l = eddl.Dense(l, 6 * depth * 4 * 4)
    l = eddl.ReLu(l)
    #
    z6 = eddl.Reshape(l, [6 * depth, 4, 4])
    #
    if variant == 'b':
        x6 = eddl.Conv(x6, 6 * depth, [3, 3], [1, 1], "same")
        x6 = eddl.ReLu(x6)
        x6 = eddl.Conv(x6, 6 * depth, [3, 3], [1, 1], "same")
        x6 = eddl.ReLu(x6)
    #
    z6 = eddl.Concat([z6, x6])
    #
    z6 = eddl.Conv(z6, 6 * depth, [3, 3], [1, 1], "same") # 4 x 4
    z6 = eddl.Conv(z6, 6 * depth, [3, 3], [1, 1], "same")
    z6 = eddl.ReLu(z6)
    #
    z5 = eddl.UpSampling(z6, [2, 2])
    #
    if variant == 'b':
        x5 = eddl.Conv(x5, 5 * depth, [3, 3], [1, 1], "same")
        x5 = eddl.ReLu(x5)
        x5 = eddl.Conv(x5, 5 * depth, [3, 3], [1, 1], "same")
        x5 = eddl.ReLu(x5)
    #
    z5 = eddl.Concat([z5, x5])
    #
    z5 = eddl.Conv(z5, 5 * depth, [3, 3], [1, 1], "same") # 8 x 8
    z5 = eddl.Conv(z5, 5 * depth, [3, 3], [1, 1], "same")
    z5 = eddl.ReLu(z5)
    #
    z4 = eddl.UpSampling(z5, [2, 2])
    #
    if variant == 'b':
        x4 = eddl.Conv(x4, 4 * depth, [3, 3], [1, 1], "same")
        x4 = eddl.ReLu(x4)
        x4 = eddl.Conv(x4, 4 * depth, [3, 3], [1, 1], "same")
        x4 = eddl.ReLu(x4)
    #
    z4 = eddl.Concat([z4, x4])
    #
    z4 = eddl.Conv(z4, 4 * depth, [3, 3], [1, 1], "same") # 16 x 16
    z4 = eddl.Conv(z4, 4 * depth, [3, 3], [1, 1], "same")
    z4 = eddl.ReLu(z4)
    #
    z3 = eddl.UpSampling(z4, [2, 2])
    #
    if variant == 'b':
        x3 = eddl.Conv(x3, 3 * depth, [3, 3], [1, 1], "same")
        x3 = eddl.ReLu(x3)
        x3 = eddl.Conv(x3, 3 * depth, [3, 3], [1, 1], "same")
        x3 = eddl.ReLu(x3)
    #
    z3 = eddl.Concat([z3, x3])
    #
    z3 = eddl.Conv(z3, 3 * depth, [3, 3], [1, 1], "same") # 32 x 32
    z3 = eddl.Conv(z3, 3 * depth, [3, 3], [1, 1], "same")
    z3 = eddl.ReLu(z3)
    #
    z2 = eddl.UpSampling(z3, [2, 2])
    #
    if variant == 'b':
        x2 = eddl.Conv(x2, 2 * depth, [3, 3], [1, 1], "same")
        x2 = eddl.ReLu(x2)
        x2 = eddl.Conv(x2, 2 * depth, [3, 3], [1, 1], "same")
        x2 = eddl.ReLu(x2)
    #
    z2 = eddl.Concat([z2, x2])
    #
    z2 = eddl.Conv(z2, 2 * depth, [3, 3], [1, 1], "same") # 64 x 64
    z2 = eddl.Conv(z2, 2 * depth, [3, 3], [1, 1], "same")
    z2 = eddl.ReLu(z2)
    #
    z1 = eddl.UpSampling(z2, [2, 2])
    #
    z1 = eddl.Conv(z1, 1 * depth, [3, 3], [1, 1], "same") # 128 x 128
    z1 = eddl.Conv(z1, 1 * depth, [3, 3], [1, 1], "same")

    if with_sigmoid:
        z1 = eddl.Conv(z1, 1, [1, 1]) # before 1 channel
        out = eddl.Sigmoid(z1)
    else:
        z1 = eddl.Conv(z1, 2, [1, 1])
        out = eddl.Softmax(z1, axis = 1)

    return out
    # ----------------------------------------------------------------

# Function for choosing a U-Net architecture variant 2

# Input: Layer x
def UNetWithPadding_2(x, with_sigmoid = False):
    # Hidden layers
    depth=32  

    x1 = x

    # The u-net network computation network is constructed
    x1 = eddl.Conv(x1, 1 * depth, [3, 3], [1, 1], "same")  # 128 x 128
    x1 = eddl.Conv(x1, 1 * depth, [3, 3], [1, 1], "same")
    x1 = eddl.ReLu(x1)

    x2 = eddl.MaxPool(x1, [2, 2], [2, 2])

    x2 = eddl.Conv(x2, 2 * depth, [3, 3], [1, 1], "same")  # 64 x 64
    x2 = eddl.Conv(x2, 2 * depth, [3, 3], [1, 1], "same")
    x2 = eddl.ReLu(x2)

    x3 = eddl.MaxPool(x2, [2, 2], [2, 2])

    x3 = eddl.Conv(x3, 3 * depth, [3, 3], [1, 1], "same") # 32 x 32
    x3 = eddl.Conv(x3, 3 * depth, [3, 3], [1, 1], "same")
    x3 = eddl.ReLu(x3)

    x4 = eddl.MaxPool(x3, [2, 2], [2, 2])

    x4 = eddl.Conv(x4, 4 * depth, [3, 3], [1, 1], "same") # 16 x 16
    x4 = eddl.Conv(x4, 4 * depth, [3, 3], [1, 1], "same")
    x4 = eddl.ReLu(x4)

    x5 = eddl.MaxPool(x4, [2, 2], [2, 2])

    x5 = eddl.Conv(x5, 5 * depth, [3, 3], [1, 1], "same") # 8 x 8
    x5 = eddl.Conv(x5, 5 * depth, [3, 3], [1, 1], "same")
    x5 = eddl.ReLu(x5)

    x6 = eddl.MaxPool(x5, [2, 2], [2, 2])

    x6 = eddl.Conv(x6, 6 * depth, [3, 3], [1, 1], "same") # 4 x 4
    x6 = eddl.Conv(x6, 6 * depth, [3, 3], [1, 1], "same")
    x6 = eddl.ReLu(x6)

    #x7 = eddl.MaxPool(x7, [2, 2], [2, 2]) # 3 x 3
    x7 = x6

    neck = eddl.Flatten(x7)

    l = neck
    l = eddl.Dense(l, 6 * depth * 4 * 4)
    l = eddl.ReLu(l)
    l = eddl.Dense(l, 6 * depth * 4 * 4)
    l = eddl.ReLu(l)

    z7 = eddl.Reshape(l, [6 * depth, 4, 4])

    z6 = z7

    z6 = eddl.Conv(z6, 6 * depth, [3, 3], [1, 1], "same") # 4 x 4
    z6 = eddl.Conv(z6, 6 * depth, [3, 3], [1, 1], "same")
    z6 = eddl.ReLu(z6)

    z5 = eddl.UpSampling(z6, [2, 2])

    z5 = eddl.Conv(z5, 5 * depth, [3, 3], [1, 1], "same") # 8 x 8
    z5 = eddl.Conv(z5, 5 * depth, [3, 3], [1, 1], "same")
    z5 = eddl.ReLu(z5)

    z4 = eddl.UpSampling(z5, [2, 2])

    z4 = eddl.Conv(z4, 4 * depth, [3, 3], [1, 1], "same") # 16 x 16
    z4 = eddl.Conv(z4, 4 * depth, [3, 3], [1, 1], "same")
    z4 = eddl.ReLu(z4)

    z3 = eddl.UpSampling(z4, [2, 2])

    z3 = eddl.Conv(z3, 3 * depth, [3, 3], [1, 1], "same") # 32 x 32
    z3 = eddl.Conv(z3, 3 * depth, [3, 3], [1, 1], "same")
    z3 = eddl.ReLu(z3)

    z2 = eddl.UpSampling(z3, [2, 2])

    z2 = eddl.Conv(z2, 2 * depth, [3, 3], [1, 1], "same") # 64 x 64
    z2 = eddl.Conv(z2, 2 * depth, [3, 3], [1, 1], "same")
    z2 = eddl.ReLu(z2)

    z1 = eddl.UpSampling(z2, [2, 2])

    z1 = eddl.Conv(z1, 1 * depth, [3, 3], [1, 1], "same") # 128 x 128
    z1 = eddl.Conv(z1, 1 * depth, [3, 3], [1, 1], "same")
    #z1 = eddl.ReLu(z1)

    if with_sigmoid:
        z1 = eddl.Conv(z1, 1, [1, 1]) # before 1 channel
        out = eddl.Sigmoid(z1)
    else:
        z1 = eddl.Conv(z1, 2, [1, 1])
        out = eddl.Softmax(z1, axis = 1)

    return out
    # ----------------------------------------------------------------

# Function for choosing a U-Net architecture variant 3
# Input: Layer x
def UNetWithPadding_3(x, with_sigmoid = False):
    # Hidden layers
    depth = 64

    x1 = x

    # The u-net network computation network is constructed.
    x1 = eddl.Conv(x1, 1 * depth, [3, 3], [1, 1], "same")  # 128 x 128
    x1 = eddl.Conv(x1, 1 * depth, [3, 3], [1, 1], "same")
    x1 = eddl.ReLu(x1)

    x2 = eddl.AvgPool(x1, [2, 2], [2, 2])

    x2 = eddl.Conv(x2, 2 * depth, [3, 3], [1, 1], "same")  # 64 x 64
    x2 = eddl.Conv(x2, 2 * depth, [3, 3], [1, 1], "same")
    x2 = eddl.ReLu(x2)

    '''
    x3 = eddl.MaxPool(x2, [2, 2], [2, 2])

    x3 = eddl.Conv(x3, 3 * depth, [3, 3], [1, 1], "same") # 32 x 32
    x3 = eddl.Conv(x3, 3 * depth, [3, 3], [1, 1], "same")
    x3 = eddl.ReLu(x3)

    x4 = eddl.MaxPool(x3, [2, 2], [2, 2])

    x4 = eddl.Conv(x4, 4 * depth, [3, 3], [1, 1], "same") # 16 x 16
    x4 = eddl.Conv(x4, 4 * depth, [3, 3], [1, 1], "same")
    x4 = eddl.ReLu(x4)

    x5 = eddl.MaxPool(x4, [2, 2], [2, 2])

    x5 = eddl.Conv(x5, 5 * depth, [3, 3], [1, 1], "same") # 8 x 8
    x5 = eddl.Conv(x5, 5 * depth, [3, 3], [1, 1], "same")
    x5 = eddl.ReLu(x5)

    x6 = eddl.MaxPool(x5, [2, 2], [2, 2])

    x6 = eddl.Conv(x6, 6 * depth, [3, 3], [1, 1], "same") # 4 x 4
    x6 = eddl.Conv(x6, 6 * depth, [3, 3], [1, 1], "same")
    x6 = eddl.ReLu(x6)

    #x7 = eddl.MaxPool(x7, [2, 2], [2, 2]) # 3 x 3
    x7 = x6

    neck = eddl.Flatten(x7)

    l = neck
    l = eddl.Dense(l, 6 * depth * 4 * 4)
    l = eddl.ReLu(l)
    l = eddl.Dense(l, 6 * depth * 4 * 4)
    l = eddl.ReLu(l)

    z7 = eddl.Reshape(l, [6 * depth, 4, 4])

    z6 = z7

    z6 = eddl.Conv(z6, 6 * depth, [3, 3], [1, 1], "same") # 4 x 4
    z6 = eddl.Conv(z6, 6 * depth, [3, 3], [1, 1], "same")
    z6 = eddl.ReLu(z6)

    z5 = eddl.UpSampling(z6, [2, 2])

    z5 = eddl.Conv(z5, 5 * depth, [3, 3], [1, 1], "same") # 8 x 8
    z5 = eddl.Conv(z5, 5 * depth, [3, 3], [1, 1], "same")
    z5 = eddl.ReLu(z5)

    z4 = eddl.UpSampling(z5, [2, 2])

    z4 = eddl.Conv(z4, 4 * depth, [3, 3], [1, 1], "same") # 16 x 16
    z4 = eddl.Conv(z4, 4 * depth, [3, 3], [1, 1], "same")
    z4 = eddl.ReLu(z4)

    z3 = eddl.UpSampling(z4, [2, 2])

    z3 = eddl.Conv(z3, 3 * depth, [3, 3], [1, 1], "same") # 32 x 32
    z3 = eddl.Conv(z3, 3 * depth, [3, 3], [1, 1], "same")
    z3 = eddl.ReLu(z3)

    z2 = eddl.UpSampling(z3, [2, 2])
    '''

    z2 = x2

    z2 = eddl.Conv(z2, 2 * depth, [3, 3], [1, 1], "same") # 64 x 64
    z2 = eddl.Conv(z2, 2 * depth, [3, 3], [1, 1], "same")
    z2 = eddl.ReLu(z2)

    z1 = eddl.UpSampling(z2, [2, 2])
    #z1 = eddl.Resize(z2, [2 * depth, 128, 128], da_mode = 'constant', coordinate_transformation_mode = 'asymmetric')

    z1 = eddl.Conv(z1, 1 * depth, [3, 3], [1, 1], "same") # 128 x 128
    z1 = eddl.Conv(z1, 1 * depth, [3, 3], [1, 1], "same")
    #z1 = eddl.ReLu(z1)

    if with_sigmoid:
        z1 = eddl.Conv(z1, 1, [1, 1]) # before 1 channel
        out = eddl.Sigmoid(z1)
    else:
        z1 = eddl.Conv(z1, 2, [1, 1])
        out = eddl.Softmax(z1, axis = 1)

    return out
    # ----------------------------------------------------------------

if __name__ == "__main__":
    # execute only if run as a script

    # Configuration
    base_data_dir = 'data/paip2020/training'
    output_dir = None
    with_sigmoid = True
    model_id = None
    gpu_mask = [1]
    models_dir = 'models.sigmoid'
    log_dir = 'log.sigmoid'
    batch_size = 60
    task = 'training'
    starting_epoch = -1
    model_filename = None
    #model_filename = f'models.softmax/unet{model_id}-{starting_epoch}.onnx'
    #model_filename = f'models/unet{model_id}-{starting_epoch}.onnx'
    threshold = 0.5

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--gpu':
            gpu_mask = [int(x) for x in sys.argv[i+1].split(sep = ',')]
        elif sys.argv[i] == '--softmax':
            with_sigmoid = False
            models_dir = 'models.softmax'
            log_dir = 'log.softmax'
        elif sys.argv[i] == '--sigmoid':
            with_sigmoid = True
            models_dir = 'models.sigmoid'
            log_dir = 'log.sigmoid'
        elif sys.argv[i] == '--batch-size':
            batch_size = int(sys.argv[i+1])
        elif sys.argv[i] == '--threshold':
            threshold = float(sys.argv[i+1])
        elif sys.argv[i] == '--model-id':
            model_id = str(sys.argv[i+1])
        elif sys.argv[i] == '--task':
            task = str(sys.argv[i+1])
        elif sys.argv[i] == '--model-filename':
            model_filename = str(sys.argv[i+1])
        elif sys.argv[i] == '--starting-epoch':
            starting_epoch = int(sys.argv[i+1])
        elif sys.argv[i] == '--data-dir':
            base_data_dir = str(sys.argv[i+1])
        elif sys.argv[i] == '--output-dir':
            output_dir = str(sys.argv[i+1])

    if task == 'training':
        #Red para Data Augmentation
        #in1: Imagen png
        in1 = eddl.Input([4, 128, 128])
        #in2: Máscara
        if with_sigmoid:
            in2 = eddl.Input([1, 128, 128]) # before 1 channel
        else:
            in2 = eddl.Input([2, 128, 128])
        #Se concatena la imagen png con la mascara
        l = eddl.Concat([in1, in2]);       
        l = eddl.RandomCropScale(l, [0.9, 1.0]) # Random Crop and Scale to orig size
        #l = eddl.CenteredCrop(l, [128, 128])  # Crop to work with sizes power 2
        img = eddl.Select(l, ["0:4"]) # UnCat [0-3] image
        if with_sigmoid:
            mask = eddl.Select(l, ["4"])  # UnCat [4] mask
        else:
            mask = eddl.Select(l, ["4:6"])  # UnCat [4-5] mask
        # Both, image and mask, have the same augmentation

        # Define DA model inputs
        danet = eddl.Model([in1, in2],[])


        # Build model for DA
        eddl.build(danet, lo = None, me = None, cs = eddl.CS_GPU(gpu_mask, mem = "full_mem")) # only in GPU 0 with low_mem setup
        eddl.summary(danet)
    else:
        danet = None

    ###############################
    # Build SegNet
    #
    if model_filename is not None:
        segnet = eddl.import_net_from_onnx_file(model_filename, mem = 0, log_level = eddl.LOG_LEVEL.ERROR)
    elif task == 'training':
        in_ = eddl.Input([4, 128, 128])
        if model_id == '1a':
            out = UNetWithPadding_1(in_, with_sigmoid = with_sigmoid, variant = 'a')
        elif model_id == '1b':
            out = UNetWithPadding_1(in_, with_sigmoid = with_sigmoid, variant = 'b')
        elif model_id == '2':
            out = UNetWithPadding_2(in_, with_sigmoid = with_sigmoid)
        elif model_id == '3':
            out = UNetWithPadding_3(in_, with_sigmoid = with_sigmoid)
        else:
            raise Exception('Unexpected model!')
        segnet = eddl.Model([in_],[out])
    else:
        raise Exception('A trained model must be provided to do inference!')

    if with_sigmoid:
        losses = ["mse"]
        optimizer = eddl.adam(1.0e-5)
        #optimizer = eddl.sgd(1.0e-5) # Optimizer
        #optimizer = eddl.rmsprop(1.0e-4) # Optimizer
    else:
        losses = ["softmax_cross_entropy"]
        #losses = ["dice"]
        #losses = ["mse"]
        optimizer = eddl.adam(1.0e-6)
        #optimizer = eddl.sgd(1.0e-8) # Optimizer
        #optimizer = eddl.rmsprop(1.0e-2) # Optimizer

    eddl.build(segnet,
          optimizer,
          losses, # Losses
          ["dice"], #, "mse"], # Metrics
          eddl.CS_GPU(gpu_mask, mem = "full_mem", lsb = 50 if sum(gpu_mask) > 1 else 1),
          #eddl.CS_CPU(4),
          init_weights = (model_filename is None)
    )
    # Train on multi-gpu with sync weights every 100 batches:
    #toGPU(segnet,[1],100,"low_mem"); # In two gpus, syncronize every 100 batches, low_mem setup
    eddl.summary(segnet)
    print('\n\n\n', flush = True)
    eddl.plot(segnet, "segnet.pdf")


    ###############################


    ###############################
    # Training
    starting_epoch += 1
    epochs = 500
    #suffix = '_mini'
    suffix = '.128x128'
    dg = DataGenerator(folder_png = f'{base_data_dir}/png_img_l3{suffix}/',
                       folder_mask = f'{base_data_dir}/mask_img_l3{suffix}/',
                       batch_size = batch_size,
                       shuffle = (task == 'training'),
                       verbose = False,
                       use_cache = False,
                       for_softmax = not with_sigmoid,
                       return_filenames = True)
    print('\n\n\n', flush = True)
    #
    if task == 'training':
        #
        for epoch in range(starting_epoch, epochs):
            #
            eddl.reset_loss(segnet)
            #
            for j in range(len(dg)):
                #
                x, y, _ = dg[j]
                x = Tensor.fromarray(x)
                y = Tensor.fromarray(y)             
                # DA
                eddl.forward(danet, [x, y])
                #
                # get COPIES of tensors from DA
                x_da = eddl.getOutput(img)
                y_da = eddl.getOutput(mask)
                #
                # SegNet
                eddl.train_batch(segnet, [x_da], [y_da])

                #eddl.print_loss(segnet, j + 1)
                print(j + 1,  'loss:', eddl.get_losses(segnet), 'metrics:', eddl.get_metrics(segnet), end = '                                          \r', flush = True)

                del x_da
                del y_da

                '''
                # We should use "mult_(255.0f)" but with normalize we can stretch its contrast and see results faster
                yout2 = eddl.getOutput(out)
                yout2 = yout2.select({"0"})
                yout2.normalize_(0.0, 255.0)
                yout2.save("./out.jpg")
                #delete yout2
                '''
            print('\n')
            dg.on_epoch_end()
            eddl.save_net_to_onnx_file(segnet, f'{models_dir}/unet{model_id}-{epoch}.onnx')

            f = open(f'{log_dir}/unet.txt', 'at')
            f.write(f'epoch {epoch} ')
            l = eddl.get_losses(segnet)
            f.write(' loss ' + ' '.join('{:g}'.format(_) for _ in l))
            m = eddl.get_metrics(segnet)
            f.write(' metric ' + ' '.join('{:g}'.format(_) for _ in m))
            f.write('\n')
            f.close()
        #
    elif task == 'evaluate':
        #

        from join_mask_2 import compute_iou

        total_iou = 0.
        counter = 0

        if output_dir is None:
            raise Exception('output dir not specified!')

        for j in range(len(dg)):
            #
            x, y, filenames = dg[j]
            x = Tensor.fromarray(x)
            #y = Tensor.fromarray(y)             
            # SegNet
            y_pred = eddl.predict(segnet, [x])
            y_pred = y_pred[0].getdata()

            assert len(y) == len(y_pred)

            if with_sigmoid:
                # use y1[0] and y2[0] because despite there is only one channel,
                # the channel is the first of 3 dimensions
                for y1, y2, filename in zip(y, y_pred, filenames):
                    if (y1[0] == 1).sum() > 0  or  (y2[0] >= threshold).sum() > 0:
                        #print((y1[0] == 1).sum(), (y2[0] >= threshold).sum())
                        v = compute_iou(a = y1[0], b = y2[0], threshold = threshold)
                        total_iou += v
                        counter += 1
                
                    #print(y1[0].shape, y2[0].shape, flush = True)
                    cv2.imwrite(output_dir + filename + '.png', (y2[0] * 255).astype(int))
            else:
                for y1, y2, filename in zip(y, y_pred, filenames):
                    if (y1[1] == 1).sum() > 0  or  (y2[1] >= threshold).sum() > 0:
                        #print((y1[1] == 1).sum(), (y2[1] >= threshold).sum())
                        v = compute_iou(a = y1[1], b = y2[1], threshold = threshold)
                        total_iou += v
                        counter += 1

                    cv2.imwrite(output_dir + filename + '.png', (y2[1] * 255).astype(int))

            print('  ', counter, total_iou / max(1, counter), end = '               \n')

            '''
            # We should use "mult_(255.0f)" but with normalize we can stretch its contrast and see results faster
            yout2 = eddl.getOutput(out)
            yout2 = yout2.select({"0"})
            yout2.normalize_(0.0, 255.0)
            yout2.save("./out.jpg")
            #delete yout2
            '''
