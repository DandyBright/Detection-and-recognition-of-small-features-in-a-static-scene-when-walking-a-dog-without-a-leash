import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
from PIL import Image
import numpy as np

sets=['train', 'val', 'test']

abs_path = os.getcwd()

def normalize_arrays(tuple_of_arrays,x_max,y_max):
    sample_size = 50
    array_x, array_y = tuple_of_arrays
    array1_normalized = array_x / x_max
    array2_normalized = array_y / y_max
    point_set = np.column_stack([np.around(array2_normalized, decimals=6),np.around(array1_normalized, decimals=6)])
    if point_set.shape[0] > sample_size:
        return point_set[np.random.choice(point_set.shape[0], size=sample_size, replace=False)]
    else:
        return point_set

def label_to_txt(array, file_path, a):
    with open(file_path , 'w') as file:
        file.write(str(a))
        for row in array:
            file.write(' '+' '.join(map(str, row)) )
            
def add_to_txt(array,file_path ,a):
    with open(file_path , 'a') as file:
        file.write('\n')
        file.write(str(a))
        for row in array:
            file.write(' '+' '.join(map(str, row)) )
#多个物体写入txt

def png_to_label(pngimg_array, file_path):
    leash = np.where(pngimg_array == 1)
    dog = np.where(pngimg_array == 2)
    x_max,y_max = pngimg_array.shape
    if len(leash[0])==0:
        edges = cv2.Canny(np.uint8(pngimg_array * 255), 0, 1)
        data_point_dog = np.where(edges == 255)
        data_point_dog = normalize_arrays(data_point_dog,x_max,y_max)
        label_to_txt(data_point_dog,file_path,2)
    elif len(dog[0])==0:
        edges = cv2.Canny(np.uint8(pngimg_array * 255), 0, 1)
        data_point_leash = np.where(edges == 255)
        data_point_leash = normalize_arrays(data_point_leash,x_max,y_max)
        label_to_txt(data_point_leash,file_path,1)
    else:
        zero_pngimg_array1 = np.zeros_like(pngimg_array, dtype=np.uint8)
        zero_pngimg_array2 = np.zeros_like(pngimg_array, dtype=np.uint8)
        zero_pngimg_array1[pngimg_array == 1] = 1
        zero_pngimg_array2[pngimg_array == 2] = 2
        edges1 = cv2.Canny(np.uint8(zero_pngimg_array1 * 255), 0, 1)
        edges2 = cv2.Canny(np.uint8(zero_pngimg_array2 * 255), 0, 1)
        data_point_leash = np.where(edges1 == 255)
        data_point_dog = np.where(edges2 == 255)
        data_point_dog = normalize_arrays(data_point_dog,x_max,y_max)
        data_point_leash = normalize_arrays(data_point_leash,x_max,y_max)
        label_to_txt(data_point_leash,file_path,1)
        add_to_txt(data_point_dog,file_path,2)

def convert_annotation(image_id):
    in_file = Image.open('/project/train/src_repo/dataset/Annotations/%s.png'%( image_id))
    pngimg_array = np.array(in_file)
    out_file = '/project/train/src_repo/dataset/labels/%s.txt'%(image_id)
    png_to_label(pngimg_array,out_file)
    
for image_set in sets:
    if not os.path.exists('/project/train/src_repo/dataset/labels/'):
        os.makedirs('/project/train/src_repo/dataset/labels/')
    image_ids = open('/project/train/src_repo/dataset/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('/project/train/src_repo/dataset/%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        list_file.write('/project/train/src_repo/dataset/images/%s.jpg\n'%(image_id))
        convert_annotation(image_id)
    list_file.close()   