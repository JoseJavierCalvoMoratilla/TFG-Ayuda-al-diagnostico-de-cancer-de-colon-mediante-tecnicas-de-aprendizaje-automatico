# Alumno: Calvo Moratilla, José Javier
# Profesor: Gómez Adrian, Jon Ander
# UPV - ETSINF
# 15-06-2021
# Ayuda al diagnóstico de cáncer de colon mediante aprendizaje automático.

# Program to display the results in a graph

#Use
#Softmax 1a  python ./python/plot.py --model-id 1a --fun softmax --output ../ --best 0.782867
#Softmax 1b  python ./python/plot.py --model-id 1b --fun softmax --output ../ --best 0.777880 
#Softmax 2a  python ./python/plot.py --model-id 2 --fun softmax --output ../ --best 0.745014 
#Sigmoid 1a  python ./python/plot.py --model-id 1a --fun sigmoid --output ../ --best 0.791934 
#Sigmoid 1b  python ./python/plot.py --model-id 1b --fun sigmoid --output ../ --best 0.802630 
#Sigmoid 2   python ./python/plot.py --model-id 2 --fun sigmoid --output ../ --best 0.771651 

# Import the libraries needed
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys

# Variables
model_id = ''
fun = ''
inputs = []
output = ''
iou_tr = []
iou_val = []
iou_test = []
best = 0.0

# If don't have the 4 input arguments it raises an exception
if len(sys.argv) != 9:
    raise Exception( f' \n Unexpected number of inputs: \n \n Expected: \n --model-id \n --fun \n --output \n --best \n  ')

# Obtain info by args
for i in range(len(sys.argv)):
        if sys.argv[i] == '--model-id':
            model_id = str(sys.argv[i + 1])
        elif sys.argv[i] == '--fun':
            fun = str(sys.argv[i + 1])
        elif sys.argv[i] == '--output':
            folder_output = str(sys.argv[i + 1])
        elif sys.argv[i] == '--best':
            best = float(sys.argv[i + 1])
                  
# Creates a string that identifies the text input files and saves this in a list
inputs.append('evolution_iou_model_' + model_id +'_'+ fun +'_in_training.txt')
inputs.append('evolution_iou_model_' + model_id +'_'+ fun +'_in_validation.txt')

# Obtain iou by text training file
with open(inputs[0]) as f:
        lines = f.readlines()
        #Series to plot
        iou_tr = [float(line.split()[0]) for line in lines]    

# Obtain iou by text validation file
with open(inputs[1]) as f:
        lines = f.readlines()
        #Series to plot
        iou_val = [float(line.split()[0]) for line in lines]     


# Obtain a list of epochs range to plot        
epochs = range(len(iou_tr))  
iou_test = np.array([best for i in range(len(epochs))])

 # Create the plot
if __name__ == '__main__':   
    plt.title('Resultados, ' + fun + ', ' + model_id)
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    #plt.axis([0, epochs[-1], min(iou_tr) - 0.10, max(iou_tr) + 0.1])
    plt.axis([0, epochs[-1], 0.0, 1.0])
    plt.grid(True)
    plt.plot(epochs, iou_tr, color ='deepskyblue')
    plt.plot(epochs, iou_val, color ='darkorange')
    plt.plot(epochs, iou_test, color='black')
    plt.legend(['Train', 'Validation', 'Test'])
    plt.savefig(output + 'plot_' + model_id + '_' + fun + '.png')
    print('The plot has been created:')