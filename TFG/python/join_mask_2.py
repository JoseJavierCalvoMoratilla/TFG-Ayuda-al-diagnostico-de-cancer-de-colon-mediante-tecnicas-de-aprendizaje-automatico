# Alumno: Calvo Moratilla, José Javier
# Profesor: Gómez Adrian, Jon Ander
# UPV - ETSINF
# 10-05-2021
# Ayuda al diagnóstico de cáncer de colon mediante aprendizaje automático.

# Program that unifies the images segmented by the U-Net network to generate a mask of original size.
import os
import sys
import numpy
import glob
import re
import cv2

# Function compute_iou: 
# Calculates IoU score between two masks
# INPUTS
# input--> Mask a
# input --> Mask b
# Threshold --> Umbral
def compute_iou(a, b, threshold = 0.5):
    i = numpy.logical_and(a >= threshold, b >= threshold).sum()
    #u = numpy.logical_or( a >= threshold, b >= threshold).sum()
    #return i / max(1, u)
    u = (a >= threshold).sum() + (b >= threshold).sum()
    return 2 * i / max(1, u)

# Execute the process from script with respective input parameters
if __name__ == '__main__':

    base_dir = 'data/paip2020/training'
    #base_dir = 'data/paip2020/validation'
    #base_dir = 'data/paip2020/test'
    output_file = None

    for i in range(len(sys.argv)):
        if sys.argv[i] == '--base-dir':
            base_dir = str(sys.argv[i + 1])
        elif sys.argv[i] == '--output-file':
            output_file = str(sys.argv[i + 1])

    original_mask_dir = base_dir + '/mask_img_l3'
    #generated_mask_dir = base_dir + '/mask_img_l3_mini'
    #generated_mask_dir = base_dir + '/mask_img_l3.128x128'
    generated_mask_dir = base_dir + '/mask_img_l3_predicted.128x128'
    output_mask_dir = base_dir + '/mask_img_l3_predicted'

    list_original_masks  = sorted(glob.glob( original_mask_dir + '/*.tif'))
    #list_generated_masks = sorted(glob.glob(generated_mask_dir + '/*.tif'))
    list_generated_masks = sorted(glob.glob(generated_mask_dir + '/*.png'))

    total_iou = 0.

    for original_filename in list_original_masks:
        original = cv2.imread(original_filename, cv2.IMREAD_UNCHANGED)
        reconstructed = numpy.zeros(original.shape)
        counters = numpy.zeros(original.shape)
        b = os.path.basename(original_filename).replace('_annotation_tumor.tif', '')
        for tile_filename in list_generated_masks:
            if b in tile_filename:
                tile = cv2.imread(tile_filename, cv2.IMREAD_UNCHANGED)
                btile = os.path.basename(tile_filename)[:-4] # basename with extension '.tif' removed
                btile = btile.split(sep = '_')
                row = int(btile[-2])
                col = int(btile[-1])
                #
                h = min(tile.shape[0], reconstructed.shape[0] - row)
                w = min(tile.shape[1], reconstructed.shape[1] - col)
                reconstructed[row : row + h, col : col + w] += tile[:h, :w]
                counters[     row : row + h, col : col + w] += 1
            #
        #
        reconstructed /= counters
        #diff = abs(reconstructed - original).sum()

        cv2.imwrite(output_mask_dir + '/' + b + '.png', reconstructed.astype(int))

        iou = compute_iou(original, reconstructed, threshold = 127)
        print(original_filename, '   ', iou)
        total_iou += iou

    print()
    print('average IoU = ', total_iou / len(list_original_masks))
    print()

    if output_file is not None:
        with open(output_file, 'at') as f:
            f.write('%.6f\n' % (total_iou / len(list_original_masks)))
            f.close()
