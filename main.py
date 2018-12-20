# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:36:37 2018

Main function to find most similar models to a given model or image

@author: AI team
"""
import argparse
import dill
import io
import numpy as np
import operator
import os
import pickle
from PIL import Image
import PIL
from random import randint

from src import extract_images as ext
from src import find_similar as fs
from src import search_catalog as sc

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras import models

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

#extract the arguments
parser = argparse.ArgumentParser(description='Extarct images, run hyperperameter search, fit the classifier, evalute the accuracy or predict the class')
parser.add_argument('--task', type=str, default='pass',
                    help="""
                    task to perform: 
                    extract_images --> extract and save images from contents.mediadecathlon.com
                    fit --> compute features for all images in the dataset
                    show_similar_products --> show products most similar to a given one
                    search_products --> find the products most similar to a given image
                    """)
parser.add_argument('--data_augmentation', type=int, default=1,
                    help="""
                    If we want (1) or not (0) to consider data augmentation (lr/ud flipping and 90,180,270 degree rotations)
                    """)
parser.add_argument('--dpt_number', type=int, default=0,
                    help="""
                    Number of the department from which we want to consider the products. Department = 0 indicates that we consider
                    the products of all departments
                    """)
parser.add_argument('--domain_id', type=str, default='0341',
                    help="""
                    Id of the domain (for instance, decathlon.ca), used to extract the Pixl IDs of the relevant products
                    """)
parser.add_argument('--product_id', type=int, default=None,
                    help="""
                    ID of the product for which we want to get the most similar products in the dataset
                    """)
parser.add_argument('--img', type=str, default=None,
                    help="""
                    Path to the image for which we want to identify the most similar products in the catalog
                    """)
parser.add_argument('--transfer_model', type=str, default='VGG',
                    help="""
                    Transfer learning model used to calculate similarity. The models currently supported are 
                    VGG19 ('VGG') and Inception_Resnet_V2 ('Inception_Resnet')
                    """)
parser.add_argument('--number', type=int, default=10,
                    help="""
                    Number of similar products we want to find
                    """)

args = parser.parse_args()

#verify the format of the calculate_features
if args.task not in ['extract_images', 'fit', 'show_similar_products', 'search_products']:
    print('Task not supported')
    args.task = 'pass'

if args.task in ['fit', 'search_products']:    
    if args.data_augmentation == 1:
        data_augmentation=True
    elif args.data_augmentation == 0:
        data_augmentation=False
    else:
        print('dataset_augmentation argument is not 0 or 1')
        args.task = 'pass'

if not (args.dpt_number >= 0 and isinstance(args.dpt_number, int)):
    print('dpt_number has to be a positive integer')
    args.task = 'pass'
    
if not (args.number >= 0 and isinstance(args.number, int)):
    print('number has to be a positive integer')
    args.task = 'pass'

if args.product_id is not None:    
    if not (args.product_id >= 0 and isinstance(args.product_id, int)):
        print('product_id has to be a positive integer')
        args.task = 'pass'

if args.img is not None:    
    if not os.path.isfile(args.img):
        print('file path to the image does not exists')
        args.task = 'pass'
        
#function to extract the images
def extract_images():
    extracter = ext.extract_images()
    extracter.run(dpt_num_department=args.dpt_number, domain_id=args.domain_id)

#function to build the dictionary of most similar products
def fit():
    sim = fs.find_similar(dpt_num_department=args.dpt_number, domain_id=args.domain_id)
    sim.fit(k=args.number, model=args.transfer_model, 
            data_augmentation=data_augmentation, save_similar_models=True)

#function to show the most similar products    
def show_similar_items():
    #load the similar items dictionary
    path = '\\data\\trained_models\\'
    with open(currentdir + path + 'similar_products_dpt_num_department_' + str(args.dpt_number) + '_model_' + args.transfer_model + '.pickle', 'rb') as file:
        similar_items = pickle.load(file)   
        
    #if no product id is provided, randomly select one
    all_items = [i for i in similar_items]
    if args.product_id is None:      
        item = all_items[randint(0, len(all_items))]
    else:
        item = args.product_id
        
    #verify that the item is in the dictionary
    if str(item) in all_items:
        #print the results
        for i in range(len(similar_items[str(item)])):
            print('Most similar product number', i+1, ': ' + similar_items[str(item)][i])
            
    else:
        print('Product number', item, 'not recognized')
        
#function to find products most similar to an image
def search_products(img):
    search = sc.search_catalog(dpt_num_department=args.dpt_number)
    search.run(img, load_features=True, model=args.transfer_model, data_augmentation=data_augmentation) 
    #print the results
    k=0
    for i in range(len(search.similar_models)):
        if search.similar_models[i] not in search.similar_models[:i]:#we remove duplicates
            k+=1
            print('Most similar product number', k, ': ' + str(search.similar_models[i]))
            if k+1 == args.number:
                break

#run the proper function given the --task argument passed to the function
if args.task == 'extract_images':
    extract_images()
    
elif args.task == 'fit':
    fit()
    
elif args.task == 'show_similar_products':
    show_similar_items()
    
elif args.task == 'search_products':
    search_products(args.img)
    