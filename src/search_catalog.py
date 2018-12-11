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
from sklearn.neighbors import NearestNeighbors
import sqlite3
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.vgg19 import preprocess_input as ppVGG19
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.inception_resnet_v2 import preprocess_input as ppIR
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.layers import GlobalAveragePooling2D

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import utils

sqlite3.register_adapter(np.ndarray, utils.adapt_array)
sqlite3.register_converter("array", utils.convert_array)

class search_catalog():
    
    def __init__(self, dpt_num_department=0):
        self.dpt_num_department = dpt_num_department
        self.model = None
    
    #method to load VGG19 and Inception_Resnet models
    def _load_model(self):
        print("Loading VGG19 pre-trained model...")
        base_model = VGG19(weights='imagenet')
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        self.VGG_model = Model(inputs=base_model.input, outputs=x)
        
        #load Inception_Resnet model
        print("Loading Inception_Resnet_V2 pre-trained model...")
        base_model = InceptionResNetV2(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        self.IR_model = Model(inputs=base_model.input, outputs=x)
    
    #method to load features of each product
    def _load_features(self, model='VGG', data_augmentation=True):
        #connect to the database
        conn = sqlite3.connect(parentdir + '\\data\\database\\features.db', detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        
        #extract the features
        if data_augmentation:
            cur.execute('SELECT pixl_id, mdl_id, features_' + model + ' FROM features_' + str(self.dpt_num_department) + ' WHERE active = ?', 
                        (1,))
        else:
            cur.execute('SELECT pixl_id, mdl_id, features_' + model + ' FROM features_' + str(self.dpt_num_department) + ' WHERE active = ? AND transformation = ?', 
                        (1,''))
        
        data = cur.fetchall()
        self.features = [i[1] for i in data]
        self.images = [i[0] for i in data]
        
        conn.close()
             
    #method to fit the knn model
    def _fit_kNN(self, algorithm='brute', metric='cosine'):
        #fit kNN model
        X = np.array(self.features)
        self.kNN = NearestNeighbors(n_neighbors=np.min([50, X.shape[0]]), algorithm=algorithm, metric=metric).fit(X)
                
    #main method - identify most similar models
    def run(self, path_image, k=5, load_model=True, load_features=True,
            fit_model=True, data_augmentation=False, algorithm='brute', metric='cosine'):
        
        self.path_to_img = image
        
        #load the model
        if load_model:
            self._load_model()
            
        #load the features
        if load_features:
            if load_features:
                self._load_features(data_augmentation=data_augmentation)
            else:
                self._calculate_augmented_features(save_features=True)
        else:
            self._load_features(data_augmentation=data_augmentation)
            
        #fit the kNN model
        if fit_model:
            self._fit_kNN(algorithm=algorithm, metric=metric)
                  
        #calculate the features of the image
        if model=='Inception_Resnet':
            img = image.load_img(path_image, target_size=(299, 299)) 
            img = image.img_to_array(img)  # convert to array
            img = np.expand_dims(img, axis=0)
            img = ppIR(img)
            self.img_features = [self.IR_model.predict(img).flatten()] 
        else:
            img = image.load_img(path_image, target_size=(224, 224)) 
            img = image.img_to_array(img)  # convert to array
            img = np.expand_dims(img, axis=0)
            img = ppVGG19(img)
            self.img_features = [self.VGG_model.predict(img).flatten()] 
                     
        #find most similar images in the training set
        _, self.NN = self.kNN.kneighbors(self.img_features)
        
        #identify most similar models
        similar_models = [self.images[i] for i in self.NN[0]][:(k+20)]
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
    search = search_catalog(dpt_num_department=0)
    search.run(image, load_features=True, data_augmentation=False)
    search.plot_similar()
        
    