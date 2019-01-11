# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:33:51 2018

Class to calculate feature vectors, and identify most similar images.
Inspired by: https://towardsdatascience.com/building-a-similar-images-finder-without-any-training-f69c0db900b5

@author: AI team
"""
import collections
import io
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
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.features = []
        self.similar_images = {}
        self.similar_items = {}
    
    #method to calculate most similar items based on a scoring system
    def _similar_items(self, k, save_similar_items=False, model='VGG'):
        #first build a img_ID:items most similar to that ID
        self.similar_img_item = {self.images[i]: [self.items[j] for j in self.NN[i][1:] if self.items[j] != self.items[i]] for i in range(len(self.images))}
        
        #loop through each item
        for itm in self.items:
            #identify all the items which had images similar to at least one image associated with the item
            sim_item = list(set([i for pid in self.item_to_img[itm] for i in self.similar_img_item[pid]]))
            
            #build an item: score dictionary, where the score is the total number of points given to each item. When an item
            #has an image similar to an image of the given item, it is given additional "similarity" points
            score = {}
            score = {m:0 for m in sim_item}
            for pid in self.item_to_img[itm]: #For all the images associated to the item...
                for i in range(len(self.similar_images[pid])): #For all the similar images to this one...
                    score[self.img_to_item[self.similar_images[pid][i]]] += k-i #Add the similarity score to each item
                                
            #order the dictionary to identify the most similar items
            sorted_by_value = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
            self.similar_items[itm] = [i[0] for i in sorted_by_value][:k]
            
        #if we want to save the similar items dictionary
        if save_similar_items:
            path = parentdir + '\\data\\trained_models\\'
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path + self.dataset + '_model_' + model + '.pickle', 'wb') as file:
                pickle.dump(self.similar_items, file, protocol=pickle.HIGHEST_PROTOCOL)
            
            print('Dictionary of similar items saved!')
            
    #method to calculate the features of every image and indicate in the database if the image is "active"
    def _calculate_features(self, data_augmentation=False):
        """
        data_augmentation: if True, we calculate the features for the augmented dataset too
        """
        
        def calculate_features(model, preprocessor, img, transformation):
            """
            transformation: type of transformation to perform data augmentation
                000: no transformation
                0001: left-right flip
                0002: up-down flip
                00090: 90d rotation
                000180: 180d rotation
                000270: 270d rotation
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
            transformations = ['000', '0001', '0002', '00090', '000180', '000270']
        else:
            transformations =['000']
                   
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
        cur.execute('CREATE TABLE IF NOT EXISTS features_' + str(self.dataset) + ' (img_id TEXT PRIMARY KEY, item_id TEXT, features_VGG array, features_Inception_Resnet array, transformation CHARACTER(20), white_background INTEGER, active INTEGER)')
        
        #create a item ID: list of associated image IDs dictionary, useful to identify most similar items after computation
        folder = parentdir + '\\data\\dataset\\' + self.dataset

        #extract the id of the items and of the images in the dataset
        self.active_items = [(i.split('_')[0], i.split('_')[1].split('.')[0]) for i in os.listdir(folder)]
        
        #remove images associated to more than one item
        images = [str(i[1]) for i in self.active_items]
        duplicates = [item for item, count in collections.Counter(images).items() if count > 1]
        images = [i for i in os.listdir(folder) if (i.split('_')[0], i.split('_')[1].split('.')[0]) in self.active_items and i.split('_')[1].split('.')[0] not in duplicates]
        items = list(set([i.split('_')[0] for i in images]))
        self.item_to_img = {items[i]:[j.split('_')[1].split('.')[0] for j in images if j.split('_')[0] == items[i]] for i in range(len(items))} #dictionary item ID: img ID
        self.img_to_item = {i.split('_')[1].split('.')[0]:i.split('_')[0] for i in images}
        
        #loop through the images, to extract their features.
        cur.execute('UPDATE features_' + str(self.dataset) + ' SET active = ?', (0,))
        ki = 0
        for i in images:
            img_ids = [i.split('_')[1].split('.')[0] + j for j in transformations]
            cur.execute('SELECT img_id, item_id FROM features_' + str(self.dataset) + ' WHERE img_id IN ({})'.format(','.join('?' * len(transformations))), 
                        img_ids)
            data=cur.fetchall()

            path = folder + '\\' + i
            img_VGG = image.load_img(path, target_size=(224, 224)) 
            img_IR = image.load_img(path, target_size=(299, 299)) 
                              
            for j in range(len(transformations)):
                #if already calculated, we activate it
                if img_ids[j] in [x[0] for x in data]:
                    cur.execute('UPDATE features_' + str(self.dataset) + ' SET active = ? WHERE img_id = ?', 
                        (1,img_ids[j]))
                
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
                    
                    #Verify color of the background (if white or not)
                    if np.array(img_VGG)[0][0][0] == 255:
                        white_background = 1
                    else:
                        white_background = 0    
                    
                    cur.execute('INSERT INTO features_' + str(self.dataset) + ' (img_id, item_id, features_VGG, features_Inception_Resnet, transformation, white_background, active) VALUES (?,?,?,?,?,?,?)', 
                            (img_ids[j], i.split('_')[0], features_VGG, features_IR, transformations[j], white_background, 1))
                                            
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
            calculate_features=True, data_augmentation=False,
            save_similar_items=True):
        """
        models currently supported: VGG (19) and Inception_Resnet (V2)
        """

        #calculate the features
        self._calculate_features(data_augmentation=data_augmentation)
        
        #connect to the datbase
        os.makedirs(parentdir + '\\data\\database', exist_ok=True)
        conn = sqlite3.connect(parentdir + '\\data\\database\\features.db', detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        
        #build the features numpy array, list of images and list of items
        cur.execute('SELECT img_id, item_id, features_' + model + ' FROM features_' + str(self.dataset) + ' WHERE active = ? AND transformation = ?', 
                        (1,'000'))
        
        data=cur.fetchall()
        self.features = [i[2] for i in data]            
        self.images = [str(i[0]).rsplit('000')[0] for i in data]
        self.items = [str(i[1]) for i in data]
        
        X = np.array(self.features)
        print('Calculating nearest neighbors')
        kNN = NearestNeighbors(n_neighbors=np.min([50, X.shape[0]]), algorithm=algorithm, metric=metric).fit(X)
        _, self.NN = kNN.kneighbors(X)
                
        #extract the similar images (of the different items) in a dictionary img ID: list of most similar img IDs
        print('Identifying similar images and items')
        self.similar_images = {self.images[i]: [self.images[j] for j in self.NN[i][1:] if self.items[j] != self.items[i]][:k] for i in range(len(self.images))}
        
        #extract the similar items in a dictionary item ID: list of most similar item IDs
        self._similar_items(k=k, save_similar_items=save_similar_items, model=model)
        
        conn.commit()
        cur.close()
        conn.close()
        
    #method to plot some example of most similar items    
    def plot_similar(self, itm=None):
        
        #if no item is provided, randomly choose one
        if itm is None:
            itm = self.items[randint(0, len(self.items))]
        
        #path to an image of this item
        folder = parentdir + '\\data\\dataset\\' + self.dataset
        path_to_img = folder + '\\' + str(itm) + '_' + str(self.item_to_img[itm][0]) + '.jpg'
        
        #path to images of models similar to this one
        path_to_similar_items = [folder + '/' + str(i) + '_' + str(self.item_to_img[i][0]) + '.jpg' for i in self.similar_items[itm]]
        
        # Create figure with sub-plots.
        utils.plot_similar(path_to_img=path_to_img, path_to_similar_items=path_to_similar_items, img_name=str(itm))
        
        
if __name__ == '__main__':
    sim = find_similar(dataset='dpt_num_department_371_domain_id_0341')
    sim.fit(k=8, model='Inception_Resnet', save_similar_items=True, data_augmentation=True)
    sim.plot_similar()