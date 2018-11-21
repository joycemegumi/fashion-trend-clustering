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
                    calculate_features --> compute features for all images in the dataset
                    find_similar_models --> find the models most similar to a given model or a given image
                    """)
parser.add_argument('--dataset_augmentation', type=int, default=1,
                    help="""
                    If we want (1) or not (0) to augment the dataset using lr/ud flipping and 90,180,270 degrees rotation
                    """)
parser.add_argument('--dpt_number', type=int, default=0,
                    help="""
                    Number of the department from which we want to consider the models. Department = 0 indicates that we want
                    images from all departments
                    """)
parser.add_argument('--model', type=int, default=None,
                    help="""
                    Number of the model for which we want to get the most similar models in the dataset
                    """)
parser.add_argument('--img', type=str, default=None,
                    help="""
                    Path to the image for which we want to identify the most similar models
                    """)

args = parser.parse_args()

#verify the format of the calculate_features
if args.task not in ['extract_images', 'calculate_features', 'find_similar_models']:
    print('Task not supported')
    args.task = 'pass'

if args.task in ['calculate_features', 'find_similar_models']:    
    if args.dataset_augmentation == 1:
        dataset_augmentation=True
    elif args.dataset_augmentation == 0:
        dataset_augmentation=False
    else:
        print('dataset_augmentation argument is not 0 or 1')
        args.task = 'pass'

if not (args.dpt_number >= 0 and isinstance(args.dpt_number, int)):
    print('dpt_number has to be a positive integer')
    args.task = 'pass'

if args.img is not None:    
    if not os.path.isfile(args.img):
        print('file path to the image does not exists')
        args.task = 'pass'
        
#function to extract the images
def extract_images():
    extracter = ext.extract_images()
    extracter.run(dpt_num_department=args.dpt_number, domain_id='0341')
    
#function to calculate the features
def calculate_features():
    #if we are considering data augmentation
    if dataset_augmentation:
        search = sc.search_catalog(dpt_num_department=args.dpt_number)
        search._load_model()
        search._calculate_augmented_features(save_features=True)
    #otherwise
    else:
        sim = fs.find_similar(dpt_num_department=args.dpt_number)
        sim.fit(save_features=True, save_similar_models=True)

#function to find models most similar to another model
def similar_to_model(model_ID):
    #load the similar model dictionary
    path = 'data\\trained_models\\'
    with open(path + 'similar_models_dpt_num_department_' + str(args.dpt_number) + '.pickle', 'rb') as file:
        similar_models = pickle.load(file)
        
    try:
        print({'similar_models': similar_models[str(model_ID)]})
    except:
        print('Model number', model_ID, 'not recognized')
        
#function to find models most similar to an image
def similar_to_img(img):
    search = sc.search_catalog(dpt_num_department=args.dpt_number)
    search.run(img, load_features=True, dataset_augmentation=dataset_augmentation) 
    print({'similar_models': search.similar_models})

#run the proper function given the --task argument passed to the function
if args.task == 'extract_images':
    extract_images()
    
elif args.task == 'calculate_features':
    calculate_features()

elif args.task == 'find_similar_models':
    if args.img is not None:
        similar_to_img(args.img)
    elif args.model is not None:
        similar_to_model(args.model)
    