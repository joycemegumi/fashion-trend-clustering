# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:54:47 2018

Class to, from the image of an, identify similar items in the dataset

@author: AI team
"""

from collections import OrderedDict, Counter
import io
import numpy as np
import os
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

#dapters to store and retrieve numpy arrays in sqlite databases...
#...see https://www.pythonforthelab.com/blog/storing-data-with-sqlite/#storing-numpy-arrays-into-databases
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

class search_catalog():
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None
    
    #method to load VGG19 and Inception_Resnet models
    def _load_model(self, model='VGG'):
        if model=='VGG':
            print("Loading VGG19 pre-trained model...")
            base_model = VGG19(weights='imagenet')
            base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            self.VGG_model = Model(inputs=base_model.input, outputs=x)
        
        if model=='Inception_Resnet':
            #load Inception_Resnet model
            print("Loading Inception_Resnet_V2 pre-trained model...")
            base_model = InceptionResNetV2(weights='imagenet', include_top=False)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            self.IR_model = Model(inputs=base_model.input, outputs=x)
    
    #method to load features of each item
    def _load_features(self, model='VGG', data_augmentation=True, remove_not_white=False):
        #connect to the database
        conn = sqlite3.connect(parentdir + '/data/database/features.db', detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        
        #extract the features
        if data_augmentation:
            if remove_not_white:
                cur.execute('SELECT img_id, item_id, features_' + model + ' FROM features_' + str(self.dataset) + ' WHERE active = ? AND white_background = ?', 
                            (1,1))
            else:
                cur.execute('SELECT img_id, item_id, features_' + model + ' FROM features_' + str(self.dataset) + ' WHERE active = ?', 
                            (1,))
                
        else:
            if remove_not_white:
                cur.execute('SELECT img_id, item_id, features_' + model + ' FROM features_' + str(self.dataset) + ' WHERE active = ? AND transformation = ? AND white_background = ?', 
                            (1,'000',1))
            else:
                cur.execute('SELECT img_id, item_id, features_' + model + ' FROM features_' + str(self.dataset) + ' WHERE active = ? AND transformation = ?', 
                            (1,'000'))
        
        data = cur.fetchall()
        self.features = [i[2] for i in data]
        self.items = [i[1] for i in data]
        self.images = [i[0].split(',000')[0] for i in data]
        
        conn.close()
             
    #method to fit the knn model
    def _fit_kNN(self, algorithm='brute', metric='cosine'):
        #fit kNN model
        X = np.array(self.features)
        self.kNN = NearestNeighbors(n_neighbors=np.min([50, X.shape[0]]), algorithm=algorithm, metric=metric).fit(X)
                
    #main method - identify most similar items
    def run(self, path_image, model='VGG', k=5, load_model=True, load_features=True,
            fit_model=True, data_augmentation=False, algorithm='brute', metric='cosine',
            nb_imgs=100, remove_not_white=False):
        
        self.path_to_img = path_image
        
        #load the model
        if load_model:
            self._load_model(model=model)
            
        #load the features
        if load_features:
            self._load_features(model=model, 
                                data_augmentation=data_augmentation,
                                remove_not_white=remove_not_white)
            
        #fit the kNN model
        if fit_model:
            self._fit_kNN(algorithm=algorithm, metric=metric)
                  
        #calculate the features of the images
        if model=='Inception_Resnet':
            img = image.load_img(path_image, target_size=(299, 299)) 
            img = image.img_to_array(img)  # convert to array
#            if remove_background:
#                img = utils.remove_background(img.astype(np.uint8))
            img = np.expand_dims(img, axis=0)
            img = ppIR(img)
            self.img_features = [self.IR_model.predict(img).flatten()] 
        else:
            img = image.load_img(path_image, target_size=(224, 224)) 
            img = image.img_to_array(img)  # convert to array
#            if remove_background:
#                img = utils.remove_background(img.astype(np.uint8))
            img = np.expand_dims(img, axis=0)
            img = ppVGG19(img)
            self.img_features = [self.VGG_model.predict(img).flatten()] 
                     
        #find most similar images in the dataset
        self.distances, self.NN = self.kNN.kneighbors(self.img_features)
        #identify most similar items

        self.similar_items = [self.items[i] for i in self.NN[0]][:nb_imgs]

        self.similar_images = [self.images[i] for i in self.NN[0]][:nb_imgs]
    
    def plot_similar(self):    
       
        path_to_similar_items = []
        for i in range(len(self.similar_images)):
            if self.similar_items[i] not in self.similar_items[:i]: #remove duplicate items
                path = [parentdir + '/data/dataset/' + str(self.dataset) + '/' + self.similar_images[i]]
                path_to_similar_items = path_to_similar_items + path
        
        # Create figure with sub-plots.
        utils.plot_similar(path_to_img=self.path_to_img, path_to_similar_items=path_to_similar_items)
        
    
if __name__=='__main__':
    image_path = parentdir + '/data/dataset/test/hockeyskatesexample2.jpg'
#    image_path = parentdir + '/data/dataset/test/usedgoaliestick_example.jpg'
#    image_path = parentdir + '/data/dataset/test/hockeystick_example.jpeg'
#    image_path = parentdir + '/data/dataset/test/hockeystick_example_2.jpg'
#    image_path = parentdir + '/data/dataset/test/hockeyskatesexample.jpeg'
    
    search = search_catalog(dataset='dpt_num_department_371_domain_id_0341')
    search.run(image_path, model='Inception_Resnet', data_augmentation=True,
               remove_not_white=True)
    search.plot_similar()
        
    
