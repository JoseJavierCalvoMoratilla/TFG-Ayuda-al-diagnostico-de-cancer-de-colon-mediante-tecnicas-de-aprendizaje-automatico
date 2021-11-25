# Alumno: Calvo Moratilla, José Javier
# Profesor: Gómez Adrian, Jon Ander
# UPV - ETSINF
# 10-02-2021
# Ayuda al diagnóstico de cáncer de colon mediante aprendizaje automático.
# Programa para la geneación de dataset a partir de imágenes médicas y su respectiva máscara.

#Importamos la librerias necesárias
import argparse
import sys
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import numpy as np
from dataGenerate import DataGenerator

USE_CONCAT = 1

MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")

#Input: Layer x
def UNetWithPadding(x, with_sigmoid = False):
    #capas ocultas
    depth=32  

    x1 = x

    #se construye el grafo de computacion de la red u-net
    x1 = eddl.LeakyReLu(eddl.Conv(x1, 1 * depth, [3, 3], [1, 1], "same"))
    x1 = eddl.LeakyReLu(eddl.Conv(x1, 1 * depth, [3, 3], [1, 1], "same"))

    x2 = eddl.MaxPool(x1, [ 2,2 ], [ 2,2 ])

    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2 * depth, [3, 3], [1, 1], "same"))
    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2 * depth, [3, 3], [1, 1], "same"))

    x3 = eddl.MaxPool(x2, [2, 2], [2, 2])

    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4 * depth, [3, 3], [1, 1], "same"))
    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4 * depth, [3, 3], [1, 1], "same"))

    x4 = eddl.MaxPool(x3, [2, 2], [2, 2])

    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8 * depth, [3, 3], [1, 1], "same"))
    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8 * depth, [3, 3], [1, 1], "same"))

    x5 = eddl.MaxPool(x4, [2, 2], [2, 2])

    x5 = eddl.LeakyReLu(eddl.Conv(x5, 8 * depth, [3, 3], [1, 1], "same"))
    x5 = eddl.LeakyReLu(eddl.Conv(x5, 8 * depth, [3, 3], [1, 1], "same"))

    x5 = eddl.Conv(eddl.UpSampling(x5, [2, 2]),  8 * depth, [3, 3], [1, 1], "same")

    if USE_CONCAT:
        x4 = eddl.Concat([x4, x5])
    else:
        x4 = eddl.Sum(x4, x5)

    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8 * depth, [3, 3], [1, 1], "same"))
    x4 = eddl.LeakyReLu(eddl.Conv(x4, 8 * depth, [3, 3], [1, 1], "same"))
    x4 = eddl.UpSampling(x4, [2, 2])
    x4 = eddl.Conv(x4, 4 * depth, [3, 3], [1, 1], "same")

    if USE_CONCAT:
        x3 = eddl.Concat([x3, x4])
    else:
        x3 = eddl.Sum(x3, x4)

    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4 * depth, [3, 3], [1, 1], "same"))
    x3 = eddl.LeakyReLu(eddl.Conv(x3, 4 * depth, [3, 3], [1, 1], "same"))
    x3 = eddl.UpSampling(x3, [2, 2])
    x3 = eddl.Conv(x3, 2 * depth, [3, 3], [1, 1], "same")

    if USE_CONCAT:
        x2 = eddl.Concat([x2,x3])
    else:
        x2 = eddl.Sum(x2,x3)

    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2 * depth, [3, 3], [1, 1], "same"))
    x2 = eddl.LeakyReLu(eddl.Conv(x2, 2 * depth, [3, 3], [1, 1], "same"))
    x2 = eddl.UpSampling(x2, [2, 2])
    x2 = eddl.Conv(x2, 1 * depth, [3, 3], [1, 1], "same")

    if USE_CONCAT:
        x1 = eddl.Concat([x1, x2])
    else:
        x1 = eddl.Sum(x1, x2)

    x1 = eddl.LeakyReLu(eddl.Conv(x1, 1 * depth, [3, 3], [1, 1], "same"))
    x1 = eddl.LeakyReLu(eddl.Conv(x1, 1 * depth, [3, 3], [1, 1], "same"))

    if with_sigmoid:
        x1 = eddl.Conv(x1, 2, [1, 1]) # before 1 channel
        out = eddl.Sigmoid(x1)
    else:
        x1 = eddl.Conv(x1, 2, [1, 1])
        out = eddl.Softmax(x1, axis = 1)

    return out

if __name__ == "__main__":
    # execute only if run as a script
    #Necesitamos preaprar el dataset en una matriz tensor?

    #Configuración
    with_sigmoid = False
    gpu_mask = [1, 1]

    #Red para Data Augmentation
    #in1: Imagen png
    in1 = eddl.Input([4, 200, 200])
    #in2: Máscara
    if with_sigmoid:
        in2 = eddl.Input([2, 200, 200]) # before 1 channel
    else:
        in2 = eddl.Input([2, 200, 200])
    #Se concatena la imagen png con la mascara
    l = eddl.Concat([in1, in2]);       
    l = eddl.RandomCropScale(l, [0.9, 1.0]) # Random Crop and Scale to orig size
    l = eddl.CenteredCrop(l, [176, 176])  # Crop to work with sizes power 2
    img = eddl.Select(l, ["0:4"]) # UnCat [0-3] image
    if with_sigmoid:
        mask = eddl.Select(l, ["4:6"])  # UnCat [4] mask
    else:
        mask = eddl.Select(l, ["4:6"])  # UnCat [4-5] mask
    # Both, image and mask, have the same augmentation

    # Define DA model inputs
    danet = eddl.Model([in1, in2],[])


    # Build model for DA
    eddl.build(danet, lo = None, me = None, cs = eddl.CS_GPU(gpu_mask, mem = "full_mem")) # only in GPU 0 with low_mem setup
    eddl.summary(danet)

    ###############################
    # Build SegNet
    #model_filename = None
    starting_epoch = 99
    model_filename = f'models.softmax/unet1-{starting_epoch}.onnx'
    #
    if model_filename is not None:
        segnet = eddl.import_net_from_onnx_file(model_filename, mem = 0, log_level = eddl.LOG_LEVEL.ERROR)
    else:
        in_ = eddl.Input([4, 176, 176])
        out = UNetWithPadding(in_, with_sigmoid = with_sigmoid)
        segnet = eddl.Model([in_],[out])

    if with_sigmoid:
        losses = ["mse"]
        optimizer = eddl.adam(1.0e-4)
        #optimizer = eddl.sgd(1.0e-5) # Optimizer
        #optimizer = eddl.rmsprop(1.0e-4) # Optimizer
    else:
        losses = ["softmax_cross_entropy"]
        #optimizer = eddl.adam(1.0e-7) # OK
        optimizer = eddl.adam(1.0e-6)
        #optimizer = eddl.sgd(1.0e-6) # Optimizer
        #optimizer = eddl.rmsprop(1.0e-6) # Optimizer

    eddl.build(segnet,
          optimizer,
          losses, # Losses
          ["mse"], # Metrics
          eddl.CS_GPU(gpu_mask, mem = "full_mem", lsb = 50),
          init_weights = (model_filename is None)
    )
    # Train on multi-gpu with sync weights every 100 batches:
    #toGPU(segnet,[1],100,"low_mem"); # In two gpus, syncronize every 100 batches, low_mem setup
    eddl.summary(segnet)
    eddl.plot(segnet, "segnet.pdf")


    ###############################


    ###############################
    # Training
    model_id = 1
    starting_epoch += 1
    epochs = 500
    base_data_dir = './data/paip2020/training'
    batch_size = 20
    dg = DataGenerator(folder_png = f'{base_data_dir}/png_img_l3_mini/',
                       folder_mask = f'{base_data_dir}/mask_img_l3_mini/',
                       batch_size = batch_size,
                       shuffle = True,
                       verbose = False,
                       for_softmax = not with_sigmoid)
    #
    for epoch in range(starting_epoch, epochs):
        eddl.reset_loss(segnet)
        for j in range(len(dg)):
            x, y, = dg[j]

            x = Tensor.fromarray(x)
            y = Tensor.fromarray(y)             

            # DA
            eddl.forward(danet, [x, y])

            # get COPIES of tensors from DA
            x_da = eddl.getOutput(img)
            y_da = eddl.getOutput(mask)

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
        eddl.save_net_to_onnx_file(segnet, f'models/unet{model_id}-{epoch}.onnx')

        f = open('log/unet.txt', 'at')
        f.write(f'epoch {epoch} ')
        l = eddl.get_losses(segnet)
        f.write(' loss ' + ' '.join('{:g}'.format(_) for _ in l))
        m = eddl.get_metrics(segnet)
        f.write(' metric ' + ' '.join('{:g}'.format(_) for _ in m))
        f.write('\n')
        f.close()
