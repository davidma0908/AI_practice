import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras

from time import time

import warnings
warnings.filterwarnings('ignore')

"""
資料夾及檔案路徑變數
"""

"""放置全部資料集的資料夾"""
data_dir_path = '/data/examples/may_the_4_be_with_u/where_am_i'


target_label_file_name = 'mapping.txt'
target_label_file_path = '/'.join((data_dir_path, target_label_file_name))

train_dir = '/'.join((data_dir_path, 'train'))
testset_dir = '/'.join((data_dir_path, 'testset'))

num_classes = 15
epochs = 100
batch_size = 32

img_size = 224

def load_data(Gray2RGB=False, mean_proc=False, test_size=0.25, img_size=img_size):
    """ Load target labels """
    with open(target_label_file_path) as f:
        all_lines = [line.split(', ') for line in f.read().splitlines()]

    target_labels = dict()
    for line in all_lines:
        target_class, target_label = line
        target_labels[target_class] = target_label

    """ Create training data list """
    train_list = []
    img_paths = []
    img_labels = []
    for key in target_labels.keys():
        for img_path in glob('{}/{}/*.jpg'.format(train_dir, key)):
            train_list.append([img_path, target_labels[key]])
            img_paths.append(img_path)
            img_labels.append(target_labels[key])
               
    """ Split the list into training set and validation set """
    train_img_paths, valid_img_paths, y_train, y_valid = train_test_split(img_paths, img_labels, test_size=test_size)
    
    X_train = []
    for path in train_img_paths:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (img_size, img_size))
        
        if Gray2RGB == True:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img = img.astype(np.float32)
            
        if mean_proc == 'VGG16_ImageNet':
            img = img - np.array([123.68, 116.779, 103.939])
            img = img[:,:,::-1]  # RGB to BGR
            
        img = (img - np.min(img)) / np.max(img)
        X_train.append(img)
    X_train = np.array(X_train, dtype=np.float32)
    
    X_valid = []
    if float(test_size) != 0.:
        for path in valid_img_paths:
            img = cv2.imread(path, 0)
            img = cv2.resize(img, (img_size, img_size))

            if Gray2RGB == True:
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            
            img = img.astype(np.float32)
            
            if mean_proc == 'VGG16_ImageNet':
                img = img - np.array([123.68, 116.779, 103.939])
                img = img[:,:,::-1]  # RGB to BGR

            img = (img - np.min(img)) / np.max(img)
            X_valid.append(img)
    X_valid = np.array(X_valid, dtype=np.float32)

    if Gray2RGB == False:
        X_train = np.reshape(X_train, X_train.shape+(1,))
        X_valid = np.reshape(X_valid, X_valid.shape+(1,))
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    
    return X_train, y_train, X_valid, y_valid


testset_list = []
test_id_list = []
for img_path in glob('{}/*.jpg'.format(testset_dir)):
    testset_list.append(img_path)
    id = img_path.split('/')[-1].split('.')[0]
    test_id_list.append(id)
testset_df = pd.DataFrame({'id': test_id_list, 'path': testset_list}).sort_values(by='id')


def load_test_data(Gray2RGB=False, mean_proc=False, img_size=img_size):
    img_path_list = []
    for img_path in glob('{}/*.jpg'.format(testset_dir)):
        img_path_list.append(img_path)
    X_test = []
    X_id = []
    for path in img_path_list:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (img_size, img_size))
        
        if Gray2RGB == True:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        if mean_proc == 'VGG16_ImageNet':
            img = img - np.array([123.68, 116.779, 103.939])
            img = img[:,:,::-1]  # RGB to BGR
            
        img = (img - np.min(img)) / np.max(img)
        img_id = path.split('/')[-1].split('.')[0]
        X_test.append(img)
        X_id.append(img_id)
        
    X_test = np.array(X_test, dtype=np.float32)
    
    if Gray2RGB == False:
        X_test = np.reshape(X_test, X_test.shape+(1,))
    
    return X_test, X_id