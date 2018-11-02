# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:33:51 2018

Class to calculate feature vectors, and identify most similar images.
Inspired by: https://towardsdatascience.com/building-a-similar-images-finder-without-any-training-f69c0db900b5

@author: AI team
"""
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from random import randint
from sklearn.neighbors import NearestNeighbors
from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import utils

class find_similar():
    
    def __init__(self, dpt_num_department=371):
        self.dpt_num_department = dpt_num_department
        self.features = []
        self.similar_images = {}
        self.similar_models = {}
    
    #method to calculate most similar products based on a scoring system
    def _similar_models(self, k, save_similar_models=False):
        #first build a pixl ID:models most similar to that ID
        self.similar_pixl_model = {self.images[i].split('_')[1][:-4]: [self.images[j].split('_')[0] for j in self.NN[i][1:] if self.images[j].split('_')[0] != self.images[i].split('_')[0]] for i in range(len(self.images))}
        
        #loop through each model
        for mdl in self.models:
            #identify all the models which had images similar to at least one image associated with the model
            sim_mdl = list(set([i for pid in self.mdl_to_pixl[mdl] for i in self.similar_pixl_model[pid]]))
            
            #build a model: score dictionary, where the score is the total number of points given to each model. When a model
            #has an image similar to an image of the given model, it is given k-rank of the similar image points
            score = {}
            score = {m:0 for m in sim_mdl}
            for pid in self.mdl_to_pixl[mdl]: #For all the images associated to the model...
                for i in range(len(self.similar_images[pid])): #For all the similar images to this one...
                    score[self.pixl_to_mdl[self.similar_images[pid][i]]] += k-i #Add the similarity score to each model
                                
            #order the dictionary to identify the most similar models
            sorted_by_value = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
            self.similar_models[mdl] = [i[0] for i in sorted_by_value][:k]
            
        #if we want to save the similar models dictionary
        if save_similar_models:
            path = parentdir + '\\data\\trained_models\\'
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path + 'similar_models_dpt_num_department_' + str(self.dpt_num_department) + '.pickle', 'wb') as file:
                pickle.dump(self.similar_models, file, protocol=pickle.HIGHEST_PROTOCOL)
            
            print('Dictionary of similar models saved!')
            
    #method to calculate the features of every image
    def _extract_features(self):
        #load VGG19 model
        print("Loading VGG19 pre-trained model...")
        base_model = VGG19(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
        
        #create a model ID: list of associated pixl IDs dictionary, useful to identify most similar models after computation
        folder = parentdir + '\\data\\dataset\\' + 'dpt_num_department_' + str(self.dpt_num_department)
        #remove images associated to more than one model
        images = [i.split('_')[1][:-4] for i in os.listdir(folder)]
        self.duplicates = [item for item, count in collections.Counter(images).items() if count > 1]
        self.images = [i for i in os.listdir(folder) if i.split('_')[1][:-4] not in self.duplicates]
        self.models = list(set([i.split('_')[0] for i in self.images]))
        self.mdl_to_pixl = {self.models[i]:[j.split('_')[1][:-4] for j in self.images if j.split('_')[0] == self.models[i]] for i in range(len(self.models))} #dictionary model ID: pixl ID
        self.pixl_to_mdl = {i.split('_')[1][:-4]:i.split('_')[0] for i in self.images}
        
        #loop through the images, to extract their features
        print('Looping through the images')      
        ki = 0
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
            
            ki += 1
            if ki % 10 == 1:
                print('Features calculated for', ki, 'images')
            
    #main method, to extract features and find nearest neighbors
    def main(self, k=5, algorithm='brute', metric='cosine',
             save_features=False, save_similar_models=False):

        #extract the features
        self._extract_features()
        
        X = np.array(self.features)
        print('Calculating nearest neighbors')
        kNN = NearestNeighbors(n_neighbors=np.min([50, X.shape[0]]), algorithm=algorithm, metric=metric).fit(X)
        _, self.NN = kNN.kneighbors(X)
        
        #if we want to save the features
        if save_features:
            path = parentdir + '\\data\\trained_models\\'
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path + 'training_features_dpt_num_department_' + str(self.dpt_num_department) + '.pickle', 'wb') as file:
                pickle.dump(kNN, file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(path + 'training_images_dpt_num_department_' + str(self.dpt_num_department) + '.pickle', 'wb') as file:
                pickle.dump(self.images, file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(path + 'training_mdl_to_pixl_dpt_num_department_' + str(self.dpt_num_department) + '.pickle', 'wb') as file:
                pickle.dump(self.mdl_to_pixl, file, protocol=pickle.HIGHEST_PROTOCOL)            
            print('Features saved!')
        
        #extract the similar images (from another model) in a dictionary pixl ID: list of most similar pixl IDs
        print('Identifying similar images and models')
        self.similar_images = {self.images[i].split('_')[1][:-4]: [self.images[j].split('_')[1][:-4] for j in self.NN[i][1:] if self.images[j].split('_')[0] != self.images[i].split('_')[0]][:k] for i in range(len(self.images))}
        
        #extract the similar models in a dictionary model ID: list of most similar models ID
        self._similar_models(k=k, save_similar_models=save_similar_models)
        
    #method to plot some example of most similar products    
    def plot_similar(self, mdl=None):
        
        #if no models is provided, randomly choose one
        mdl = self.models[randint(0, len(self.models))]
        
        #path to an image of this model
        folder = parentdir + '\\data\\dataset\\' + 'dpt_num_department_' + str(self.dpt_num_department)
        path_to_img = folder + '\\' + str(mdl) + '_' + str(self.mdl_to_pixl[mdl][0]) + '.jpg'
        
        #path to images of models similar to this one
        path_to_similar_mdls = [folder + '/' + str(i) + '_' + str(self.mdl_to_pixl[i][0]) + '.jpg' for i in self.similar_models[mdl]]
        
        # Create figure with sub-plots.
        utils.plot_similar(path_to_img=path_to_img, path_to_similar_mdls=path_to_similar_mdls, img_name=str(mdl))
        self.x = path_to_similar_mdls
            
if __name__ == '__main__':
    sim = find_similar(dpt_num_department=999)
    sim.main(k=8, save_features=True)
    sim.plot_similar()