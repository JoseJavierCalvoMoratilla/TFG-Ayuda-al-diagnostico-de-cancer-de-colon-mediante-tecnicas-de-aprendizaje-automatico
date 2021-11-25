# Alumno: Calvo Moratilla, José Javier
# Profesor: Gómez Adrian, Jon Ander
# UPV - ETSINF
# 10-02-2021
# Ayuda al diagnóstico de cáncer de colon mediante aprendizaje automático.

# The dimensionality of the png images and the mask is reduced to level3.

# Original code from paip2020
# The original code did not generate the image and mask under the same conditions,
# so the code has been modified to reduce the images to a fully matching l3 level.

# use python ./python/xml2mask.py --base-data ./data/paip2020/training

# Import the libraries needed
import numpy as np
import xml.etree.ElementTree as et
import os, glob, re
from tqdm import tqdm
import tifffile, cv2
import openslide
import sys

# If don't have the 1 input arguments it raises an exception
if len(sys.argv) != 3:
    raise Exception( f' \n Unexpected number of inputs: \n \n Expected: \n --base-data \n  ')

# Obtain info by args
for i in range(len(sys.argv)):
        if sys.argv[i] == '--base-data':
            base_data_dir = str(sys.argv[i + 1])        

# The respective necessary folders are created
wsi_load_dir = base_data_dir + '/svs'
xml_load_dir = base_data_dir + '/xml'
wsi_filenames = sorted(glob.glob(wsi_load_dir + '/*.svs') + glob.glob(wsi_load_dir + '/*.SVS'))
xml_filenames = sorted(glob.glob(xml_load_dir + '/*.xml') + glob.glob(xml_load_dir + '/*.XML'))

# The level of reduction to be performed on the image is defined.
level = 3
div = 4**level ## Level0 scale to Level2 scale
div = div // 2                            

# The output folders for the customized png and tiff images are defined for each different reduction level.
png_save_dir = f'{base_data_dir}/png_img_l{level}/'
mask_save_dir = f'{base_data_dir}/mask_img_l{level}/'

# Create folders if they do not exist
os.makedirs(png_save_dir, exist_ok=True)
os.makedirs(mask_save_dir, exist_ok=True)

# A search object 're' is compiled to search for a substring in a string
q = re.compile('training_data_[0-9]{2}')

'''
Annotations (root)
> Annotation (get 'Id' -> 1: tumor area)
 > Regions
  > Region (get 'NegativeROA' -> 0: positive area // 1: inner negative area)
   > Vertices
    > Vertex (get 'X', 'Y')
'''


# xml2mask function
# Converts an xml file where the areas labeled in vectors are stored into a mask in tiff format.
# inputs:
# Name of xml
# Mask size
def xml2mask(xml_filename, shape):
  # print('reconstructing sparse xml to contours of div={}..'.format(div))
  ret = dict()
  board_pos = None
  board_neg = None
  # Annotations >> 
  e = et.parse(xml_filename).getroot()
  e = e.findall('Annotation')
  assert(len(e) == 1), len(e)
  for ann in e:
    #board_pos = np.zeros(shape[:2], dtype=np.uint8)
    #board_neg = np.zeros(shape[:2], dtype=np.uint8)
    board_pos = np.zeros([shape[1], shape[0]], dtype=np.uint8)
    board_neg = np.zeros([shape[1], shape[0]], dtype=np.uint8)
    id_num = int(ann.get('Id'))
    assert(id_num == 1)# or id_num == 2)
    regions = ann.findall('Regions')
    assert(len(regions) == 1)
    rs = regions[0].findall('Region')
    plistlist = list()
    nlistlist = list()
    print('rs:', len(rs))
    for i, r in enumerate(rs):
      ylist = list()
      xlist = list()
      plist, nlist = list(), list()
      negative_flag = int(r.get('NegativeROA'))
      assert negative_flag == 0 or negative_flag == 1
      negative_flag = bool(negative_flag)
      vs = r.findall('Vertices')[0]
      vs = vs.findall('Vertex')
      vs.append(vs[0]) # last dot should be linked to the first dot
      for v in vs:
        y, x = int(v.get('Y').split('.')[0]), int(v.get('X').split('.')[0])
        if div is not None:
          y //= div
          x //= div
        if y >= shape[1]:
          y = shape[1]-1
        elif y < 0:
          y = 0
        if x >= shape[0]:
          x = shape[0]-1
        elif x < 0:
          x = 0
        ylist.append(y)
        xlist.append(x)
        if negative_flag:
          nlist.append((x, y))         
        else:
          plist.append((x, y))
          
      if plist:
        plistlist.append(plist)
      else:
        nlistlist.append(nlist)
    for plist in plistlist:
      board_pos = cv2.drawContours(board_pos, [np.array(plist, dtype=np.int32)], -1, [255, 0, 0], -1)
    for nlist in nlistlist:
      board_neg = cv2.drawContours(board_neg, [np.array(nlist, dtype=np.int32)], -1, [255, 0, 0], -1)
    ret[id_num] = (board_pos>0) * (board_neg==0)
  return ret

# save_mask function
# Save the mask
# inputs:
# Name of xml
# Mask size
# Level of reduction
def save_mask(xml_filename, shape, level):
    wsi_id = q.findall(xml_filename)[0]
    save_filename = mask_save_dir + '/' + f'{wsi_id}_l{level}_annotation_tumor.tif'
    ret = xml2mask(xml_filename, shape)
    tifffile.imsave(save_filename, (ret[1]>0).astype(np.uint8)*255, compress=9)


# load_svs_shape function
# Obtains the dimensions of an image by performing dimensionality reduction at a specific level.
# inputs:
# Name of xml
# Mask size
# Level of reduction
def load_svs_shape(imgh, level):      
    return [imgh.level_dimensions[level][0], imgh.level_dimensions[level][1]] #canviiiiii     1 <--> 0


# save_png function
# Saves the image in a specific dimension depending on the specified level.
# inputs:
# Image input
# Name of wsi image
# Level of reduction
def save_png(img, wsi_id, level): 
    shape = load_svs_shape(img, level=level)
    newimg = img.read_region((0, 0), 3, shape)  
    save_filename = png_save_dir + '/' + f'{wsi_id}_l{level}.png'
    newimg.save(save_filename)

# Create images in png format and masks in tiff format.
if __name__ == '__main__':
    print()
    print(base_data_dir)
    print(wsi_load_dir, xml_load_dir)
    print()
    for wsi_filename, xml_filename in tqdm(zip(wsi_filenames, xml_filenames), total=len(wsi_filenames)):
        wsi_id = q.findall(wsi_filename)[0]
        xml_id = q.findall(xml_filename)[0]       
        assert wsi_id == xml_id
        img = openslide.OpenSlide(wsi_filename)
        shape = load_svs_shape(img, level)
        save_mask(xml_filename, shape, level)
        save_png(img, wsi_id, level)
        img.close()
