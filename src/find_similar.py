# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:33:51 2018

Class to calculate feature vectors, and identify most similar images.
Inspired by: https://towardsdatascience.com/building-a-similar-images-finder-without-any-training-f69c0db900b5

@author: AI team
"""
import bz2
import collections
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from random import randint
from scipy.ndimage import rotate
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


class find_similar():
    
    def __init__(self, dpt_num_department=371, domain_id='0341'):
        self.dpt_num_department = dpt_num_department
        self.domain_id = domain_id
        self.features = []
        self.similar_images = {}
        self.similar_models = {}
     
    #method to extract active models
    def _extract_IDs(self):        
        models = utils.read_S3(dpt_num_department=self.dpt_num_department,
                                    domain_id=self.domain_id)
        self.active_models = [tuple(i) for i in models[['product_id_model', 'id']].values]
    
    #method to calculate most similar products based on a scoring system
    def _similar_models(self, k, save_similar_models=False, model='VGG'):
        #first build a pixl ID:models most similar to that ID
        self.similar_pixl_model = {self.images[i]: [self.models[j] for j in self.NN[i][1:] if self.models[j] != self.models[i]] for i in range(len(self.images))}
        
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
            with open(path + 'similar_products_dpt_num_department_' + str(self.dpt_num_department) + '_model_' + model + '.pickle', 'wb') as file:
                pickle.dump(self.similar_models, file, protocol=pickle.HIGHEST_PROTOCOL)
            
            print('Dictionary of similar models saved!')
            
    #method to calculate the features of every image and indicate in the database if the image is "active"
    def _calculate_features(self, data_augmentation=False):
        """
        data_augmentation: if True, we calculate the features for the augmented dataset too
        """
        
        def calculate_features(model, preprocessor, img, transformation):
            """
            transformation: type of transformation to perform data augmentation
                0001: left-right flip
                0002: up-down flip
                00090: 90d rotation
                000180: 180d rotation
                000279: 270d rotation
                empty string: no transformation
            """                       
            #preprocess the image
            img = image.img_to_array(img)  # convert to array
            
            #flip
            if transformation=='0001':
                img = np.fliplr(img)
            elif transformation=='0002':
                img = np.flipud(img)
            #rotate
            elif transformation=='00090':
                img = rotate(img, angle=90)
            elif transformation=='000180':
                img = rotate(img, angle=180)
            elif transformation=='000270':
                img = rotate(img, angle=270)
            
            img = np.expand_dims(img, axis=0)
            
            img = preprocessor(img)
             
            return model.predict(img).flatten()
        
        if data_augmentation:
            transformations = ['', '0001', '0002', '00090', '000180', '000270']
        else:
            transformations =['']
                   
        #load VGG19 model
        print("Loading VGG19 pre-trained model...")
        base_model = VGG19(weights='imagenet')
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        VGG_model = Model(inputs=base_model.input, outputs=x)
        
        #load Inception_Resnet model
        print("Loading Inception_Resnet_V2 pre-trained model...")
        base_model = InceptionResNetV2(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        IR_model = Model(inputs=base_model.input, outputs=x)
               
        #connect to the database, and create the features table if it does not exists
        os.makedirs(parentdir + '\\data\\database', exist_ok=True)
        conn = sqlite3.connect(parentdir + '\\data\\database\\features.db', detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS features_' + str(self.dpt_num_department) + ' (pixl_id INTEGER PRIMARY KEY, mdl_id INTEGER, features_VGG array, features_Inception_Resnet array, transformation CHARACTER(20), active INTEGER)')
        
        #create a model ID: list of associated pixl IDs dictionary, useful to identify most similar models after computation
        folder = parentdir + '\\data\\dataset\\' + 'dpt_num_department_' + str(self.dpt_num_department)
        
        #remove images associated to more than one model
        self._extract_IDs()
        images = [str(i[1]) for i in self.active_models]
        duplicates = [item for item, count in collections.Counter(images).items() if count > 1]
        images = [i for i in os.listdir(folder) if (int(i.split('_')[0]), int(i.split('_')[1][:-4])) in self.active_models and i.split('_')[1][:-4] not in duplicates]
        models = list(set([i.split('_')[0] for i in images]))
        self.mdl_to_pixl = {models[i]:[j.split('_')[1][:-4] for j in images if j.split('_')[0] == models[i]] for i in range(len(models))} #dictionary model ID: pixl ID
        self.pixl_to_mdl = {i.split('_')[1][:-4]:i.split('_')[0] for i in images}
        
        #loop through the images, to extract their features.
        cur.execute('UPDATE features_' + str(self.dpt_num_department) + ' SET active = ?', (0,))
        ki = 0
        for i in images:
            pixl_ids = [int(i.split('_')[1][:-4] + j) for j in transformations]
            cur.execute('SELECT pixl_id, mdl_id FROM features_' + str(self.dpt_num_department) + ' WHERE pixl_id IN ({})'.format(','.join('?' * len(transformations))), 
                        pixl_ids)
            data=cur.fetchall()

            path = folder + '\\' + i
            img_VGG = image.load_img(path, target_size=(224, 224)) 
            img_IR = image.load_img(path, target_size=(299, 299)) 
                              
            for j in range(len(transformations)):
                #if already calculated, we activate it
                if pixl_ids[j] in [x[0] for x in data]:
                    cur.execute('UPDATE features_' + str(self.dpt_num_department) + ' SET active = ? WHERE pixl_id = ?', 
                        (1,pixl_ids[j]))
                
                #otherwise, we calculate it   
                else:                                    
                    #VGG model
                    features_VGG = calculate_features(model=VGG_model, preprocessor=ppVGG19,
                                                      img=img_VGG,
                                                      transformation=transformations[j])
                    
                    #Inception_Resnet model
                    features_IR = calculate_features(model=IR_model, preprocessor=ppIR,
                                                      img=img_IR,
                                                      transformation=transformations[j])
                    
                    cur.execute('INSERT INTO features_' + str(self.dpt_num_department) + ' (pixl_id, mdl_id, features_VGG, features_Inception_Resnet, transformation, active) VALUES (?,?,?,?,?,?)', 
                            (pixl_ids[j], i.split('_')[0], features_VGG, features_IR, transformations[j], 1))
                                            
            ki += 1
            if ki % 100 == 1:
                #commit changes
                conn.commit()
                print('Features known or calculated for', ki, 'images')
         
        conn.commit()
        cur.close()
        conn.close()
        
    #main method, to extract features and find nearest neighbors
    def fit(self, k=5, algorithm='brute', metric='cosine', model='VGG',
            calculate_features=True, data_augmentation=False, save_similar_models=True):
        """
        models currently supported VGG (19) and Inception_Resnet
        """

        #calculate the features
        self._calculate_features(data_augmentation=data_augmentation)
        
        #connect to the datbase
        os.makedirs(parentdir + '\\data\\database', exist_ok=True)
        conn = sqlite3.connect(parentdir + '\\data\\database\\features.db', detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        
        #build the features numpy array, list of images and list of models
        cur.execute('SELECT pixl_id, mdl_id, features_' + model + ' FROM features_' + str(self.dpt_num_department) + ' WHERE active = ? AND transformation = ?', 
                        (1,''))
        
        data=cur.fetchall()
        self.features = [i[2] for i in data]            
        self.images = [str(i[0]) for i in data]
        self.models = [str(i[1]) for i in data]
        
        X = np.array(self.features)
        print('Calculating nearest neighbors')
        kNN = NearestNeighbors(n_neighbors=np.min([50, X.shape[0]]), algorithm=algorithm, metric=metric).fit(X)
        _, self.NN = kNN.kneighbors(X)
                
        #extract the similar images (from another model) in a dictionary pixl ID: list of most similar pixl IDs
        print('Identifying similar images and models')
        self.similar_images = {self.images[i]: [self.images[j] for j in self.NN[i][1:] if self.models[j] != self.models[i]][:k] for i in range(len(self.images))}
        
        #extract the similar models in a dictionary model ID: list of most similar models ID
        self._similar_models(k=k, save_similar_models=save_similar_models, model=model)
        
        conn.commit()
        cur.close()
        conn.close()
        
    #method to plot some example of most similar products    
    def plot_similar(self, mdl=None):
        
        #if no models is provided, randomly choose one
        if mdl is None:
            mdl = self.models[randint(0, len(self.models))]
        
        #path to an image of this model
        folder = parentdir + '\\data\\dataset\\' + 'dpt_num_department_' + str(self.dpt_num_department)
        path_to_img = folder + '\\' + str(mdl) + '_' + str(self.mdl_to_pixl[mdl][0]) + '.jpg'
        
        #path to images of models similar to this one
        path_to_similar_mdls = [folder + '/' + str(i) + '_' + str(self.mdl_to_pixl[i][0]) + '.jpg' for i in self.similar_models[mdl]]
        
        # Create figure with sub-plots.
        utils.plot_similar(path_to_img=path_to_img, path_to_similar_mdls=path_to_similar_mdls, img_name=str(mdl))
        
        
if __name__ == '__main__':
    sim = find_similar(dpt_num_department=371)
#    sim._calculate_features()
    sim.fit(k=8, model='Inception_Resnet', save_similar_models=True, data_augmentation=True)
    sim.plot_similar()