# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:33:51 2018

Class to calculate feature vectors, and identify most similar images.
Inspired by: https://towardsdatascience.com/building-a-similar-images-finder-without-any-training-f69c0db900b5

@author: AI team
"""
import dill
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.neighbors import NearestNeighbors
from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import backend as K

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

class find_similar():
    
    def __init__(self):
        self.features = []
        self.similar_images = {}
        self.similar_models = {}
    
    #method to calculate most similar products based on a scoring system
    def _similar_models(self):
        #first build a pixl ID:models most similar to that ID
        self.similar_pixl_model = {self.images[i].split('_')[1][:-4]: [self.images[j].split('_')[0] for j in self.NN[i][1:] if self.images[j].split('_')[0] != self.images[i].split('_')[0]] for i in range(len(self.images))}
        
        #loop through each model
        for mdl in self.models:
            #identify all the models which had images similar to at least one image associated with the model
            sim_mdl = []
            #build a model: score dictionary, where the score is the total number of points given to each model. When a model
            #has an image similar to an image of the given model, it is given 1/(1+rank of the similar image) points
            
            #order the dictionary to identify the most similar models
            
    #method to calculate the features of every image
    def _extract_features(self, dpt_num_department):
        #load VGG19 model
        print("Loading VGG19 pre-trained model...")
        base_model = VGG19(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
        
        #create a model ID: list of associated pixl IDs dictionary, useful to identify most similar models after computation
        #identify all different models
        folder = parentdir + '\\data\\dataset\\' + 'dpt_num_department_' + str(dpt_num_department)
        self.images = os.listdir(folder)
        self.models = list(set([i.split('_')[1] for i in self.images]))
        self.mdl_to_pixl = {self.models[i]:[j.split('_')[1][:-4] for j in self.images if j.split('_')[0] != self.models[i]] for i in range(len(self.models))} #dictionary model ID: pixl ID
        
        #loop through the images, to extract their features
        print('Looping through the images')      
        for f in self.images:
            path = folder + '\\' + f
            #read the image
            img = image.load_img(path, target_size=(224, 224)) 
            
            #preprocess the image
            img = image.img_to_array(img)  # convert to array
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            
            #extract and flatten the features
            self.features.append(model.predict(img).flatten())
            
    #main method, to extract features and find nearest neighbors
    def main(self, dpt_num_department=371, k=5, algorithm='brute', metric='cosine'):
        #extract the features
        self._extract_features(dpt_num_department=dpt_num_department)
        
        X = np.array(self.features)
        print('Calculating nearest neighbors')
        kNN = NearestNeighbors(n_neighbors=k+20, algorithm=algorithm, metric=metric).fit(X)
        _, self.NN = kNN.kneighbors(X)
        
        #extract the similar images (from another model) in a dictionary pixl ID: list of most similar pixl IDs
        self.similar_images = {self.images[i].split('_')[1][:-4]: [self.images[j].split('_')[1][:-4] for j in self.NN[i][1:] if self.images[j].split('_')[0] != self.images[i].split('_')[0]][:k] for i in range(len(self.images))}
        
        #extract the similar models in a dictionary model ID: list of most similar models ID
        self._similar_models()
            
if __name__ == '__main__':
    sim = find_similar()
    sim.main(dpt_num_department=371)