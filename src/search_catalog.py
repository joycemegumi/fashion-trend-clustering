# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:54:47 2018

Class to, from the image of a product, identify similar products in the catalog

@author: AI team
"""

import collections
from collections import OrderedDict
import numpy as np
import os
import pickle
from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import preprocessing

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import utils

class search_catalog():
    
    def __init__(self, dpt_num_department=371):
        self.dpt_num_department = dpt_num_department
        self.model = None
    
    #method to load VGG19 model
    def _load_model(self):
        print("Loading VGG19 pre-trained model...")
        base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    
    #method to load features of each product
    def _load_features(self, dataset_augmentation=True):
        if dataset_augmentation:
            path_kNN = parentdir + '\\data\\trained_models\\training_features_augmented_'
            path_images = parentdir + '\\data\\trained_models\\training_images_augmented_'
        else:
            path_kNN = parentdir + '\\data\\trained_models\\training_features_'
            path_images = parentdir + '\\data\\trained_models\\training_images_'
            
        with open(path_kNN + 'dpt_num_department_' + str(self.dpt_num_department) + '.pickle', 'rb') as file:
            self.kNN = pickle.load(file)
        with open(path_images + 'dpt_num_department_' + str(self.dpt_num_department) + '.pickle', 'rb') as file:
            self.images = pickle.load(file)
        
    #method to calculate features for all images and augmented ones
    def _calculate_augmented_features(self, flip=True, rotate=True, algorithm='brute', metric='cosine',
                                      verbose=True, save_features=True):
        from scipy.ndimage import rotate
        from sklearn.neighbors import NearestNeighbors
        
        folder = parentdir + '\\data\\dataset\\' + 'dpt_num_department_' + str(self.dpt_num_department)
        
        #remove images associated to more than one model
        images = [i.split('_')[1][:-4] for i in os.listdir(folder)]
        self.duplicates = [item for item, count in collections.Counter(images).items() if count > 1]
        self.true_images = [i for i in os.listdir(folder) if i.split('_')[1][:-4] not in self.duplicates]
        self.images = [] #list containing the true images in addition to the augmented ones
        self.features = []
        
        #from an image to its features
        def calculate_features(img):
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            return self.model.predict(img).flatten()
        
        #loop through all the images
        print('Looping through the images')  
        ki=0
        for f in self.true_images:
            path = folder + '\\' + f
            #read the image
            img = preprocessing.image.load_img(path, target_size=(224, 224)) 
            
            #convert to an array
            img = preprocessing.image.img_to_array(img)  # convert to array

            #calculate the features for the true image
            self.images = self.images + [f]
            self.features.append(calculate_features(img))

            #calculate the features for the flipped image (left-right and up-down)
            if flip:
                self.images = self.images + [f[:-4] + '-lr.jpg']
                self.features.append(calculate_features(np.fliplr(img)))
                self.images = self.images + [f[:-4] + '-ud.jpg']
                self.features.append(calculate_features(np.flipud(img)))
            
            #calculate the features for the rotated images
            if rotate:
                self.images = self.images + [f[:-4] + '-r90.jpg']
                self.features.append(calculate_features(rotate(img, angle=90)))
                self.images = self.images + [f[:-4] + '-r180.jpg']
                self.features.append(calculate_features(rotate(img, angle=180)))
                self.images = self.images + [f[:-4] + '-r270.jpg']
                self.features.append(calculate_features(rotate(img, angle=270)))
            
            ki += 1
            if ki % 10 == 1:
                print('Features calculated for', ki, 'images')
        
        #run the knn algorithm
        X = np.array(self.features)
        print('Calculating nearest neighbors')
        self.kNN = NearestNeighbors(n_neighbors=np.min([50, X.shape[0]]), algorithm=algorithm, metric=metric).fit(X)
        
        if save_features:
            path = parentdir + '\\data\\trained_models\\'
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path + 'training_features_augmented_dpt_num_department_' + str(self.dpt_num_department) + '.pickle', 'wb') as file:
                pickle.dump(self.kNN, file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(path + 'training_images_augmented_dpt_num_department_' + str(self.dpt_num_department) + '.pickle', 'wb') as file:
                pickle.dump(self.images, file, protocol=pickle.HIGHEST_PROTOCOL)
                
            print('Features saved!')
        
    #main method - identify most similar models
    def run(self, image, k=5, load_model=True, load_features=True,
            dataset_augmentation=False):
        
        self.path_to_img = image
        
        #load the model
        if load_model:
            self._load_model()
            
        #we can either load the features, or calculate features for all images considering data augmentation
        if dataset_augmentation:
            if load_features:
                self._load_features(dataset_augmentation=dataset_augmentation)
            else:
                self._calculate_augmented_features(save_features=True)
        else:
            self._load_features(dataset_augmentation=dataset_augmentation)
                  
        #load and preprocess the image
        img = preprocessing.image.load_img(image, target_size=(224, 224))     
        img = preprocessing.image.img_to_array(img)  # convert to array
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        #calculate the features of the image
        self.img_features = [self.model.predict(img).flatten()] 
         
        #find most similar images in the training set
        _, self.NN = self.kNN.kneighbors(self.img_features)
        
        #identify most similar models
        similar_models = [self.images[i].split('_')[0] for i in self.NN[0]][:(k+20)]
        self.similar_models = list(OrderedDict.fromkeys(similar_models))[:k] #we remove duplicate models
    
    def plot_similar(self):        
        # Load the mdl to pixl ID file
        path = parentdir + '\\data\\trained_models\\'
        with open(path + 'training_mdl_to_pixl_dpt_num_department_' + str(self.dpt_num_department) + '.pickle', 'rb') as file:
                mdl_to_pixl = pickle.load(file)
                
        #find the path to an image of the similar models
        path_to_similar_mdls = [parentdir + '/data/dataset/dpt_num_department_' + str(self.dpt_num_department) + '/' + str(i) + '_' + str(mdl_to_pixl[i][0]) + '.jpg' for i in self.similar_models]
        
        # Create figure with sub-plots.
        utils.plot_similar(path_to_img=self.path_to_img, path_to_similar_mdls=path_to_similar_mdls)
        
    
if __name__=='__main__':
#    image = parentdir + '/data/dataset/test/hockey_skates_example_2.jpg'
    image = parentdir + '/data/dataset/test/used_goalie_stick_example.jpg'
#    image = parentdir + '/data/dataset/test/hockey_stick_example.jpeg'
#    image = parentdir + '/data/dataset/test/hockey_stick_example_2.jpg'
    search = search_catalog(dpt_num_department=371)
    search.run(image, load_features=True, dataset_augmentation=True)
    search.plot_similar()
        
    