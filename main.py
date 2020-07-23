# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:36:37 2018

Main function to find most similar items to a given item or image

@author: AI team
"""
import argparse
import shutil, os
import pickle
from random import randint

#from src import extract_images as ext
from src import find_similar as fs
from src import search_catalog as sc

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

#extract the arguments
parser = argparse.ArgumentParser(description='Build a dictionary of most similar items, and search for the most similar item in a dataset')
parser.add_argument('--task', type=str, default='pass',
                    help="""
                    task to perform: 
                    fit --> compute features for all the images in the dataset
                    show_similar_items --> show the items most similar to a given item_id
                    visual_search --> find the items most similar to a given image
                    """)
parser.add_argument('--data_augmentation', type=int, default=1,
                    help="""
                    If we want (1) or not (0) to consider data augmentation (lr/ud flipping and 90,180,270 degree rotations)
                    """)
parser.add_argument('--dataset', type=str, default=None,
                    help="""
                    Name of the dataset containing the image of all the items
                    """)
parser.add_argument('--item', type=str, default=None,
                    help="""
                    ID of the item for which we want to show the most similar items in the dataset
                    """)
parser.add_argument('--img', type=str, default=None,
                    help="""
                    Path to the image for which we want to identify the most similar items in the catalog
                    """)
parser.add_argument('--transfer_model', type=str, default='Inception_Resnet',
                    help="""
                    Transfer learning model used to calculate similarity. The models currently supported are 
                    VGG19 ('VGG') and Inception_Resnet_V2 ('Inception_Resnet')
                    """)
parser.add_argument('--number', type=int, default=10,
                    help="""
                    Maximum number of similar items we want to find or show
                    """)

args = parser.parse_args()

#verify the format of the arguments
if args.task not in ['fit', 'show_similar_items', 'visual_search']:
    print('task not supported')
    args.task = 'pass'

if args.task in ['fit', 'visual_search']:    
    if args.data_augmentation == 1:
        data_augmentation=True
    elif args.data_augmentation == 0:
        data_augmentation=False
    else:
        print('dataset_augmentation argument is not 0 or 1')
        args.task = 'pass'

if args.dataset is not None:
    print(currentdir + '/data/dataset/' + args.dataset)
    if not os.path.isdir(currentdir + '/data/dataset/' + args.dataset):
        print('dataset does not exists')
        args.task = 'pass'

if args.img is not None:    
    if not os.path.isfile(args.img):
        print('file path to the image does not exists')
        args.task = 'pass'

if args.transfer_model not in ['VGG', 'Inception_Resnet']:
    print('transfer_model not supported')
    args.task = 'pass'
        
#function to build the dictionary of most similar items
def fit():
    sim = fs.find_similar(dataset=args.dataset)
    sim.fit(k=args.number, model=args.transfer_model, 
            data_augmentation=data_augmentation, save_similar_items=True)

#function to show the most similar items to a given item_id 
def show_similar_items():
    #load the similar items dictionary
    path = '/data/trained_models/'
    with open(currentdir + path + str(args.dataset) + '_model_' + args.transfer_model + '.pickle', 'rb') as file:
        similar_items = pickle.load(file)   
        
    #if no item id is provided, randomly select one
    all_items = [i for i in similar_items]
    if args.item is None:      
        item = all_items[randint(0, len(all_items))]
    else:
        item = args.item

    destination = os.path.join('.', 'data', 'output', item)
    try:
        os.makedirs(destination, exist_ok=True)
    except OSError:
        print ("Creation of the directory %s failed" % destination)
    else:
        print ("Successfully created the directory %s " % destination)
        
    #verify that the item is in the dictionary
    if item in all_items:
        #print the results
        subject_source = os.path.join('.', 'data', 'dataset', 'fashion_data', item + '.jpg')
        print('Most similar items to item:', subject_source)
        #copy and move subject source to folder
        shutil.copy(subject_source, os.path.join(destination, str(0) + '-' + item + '.jpg'))
        for i in range(len(similar_items[item])):
            source = os.path.join('.', 'data', 'dataset', 'fashion_data', similar_items[item][i] + '.jpg')
            #copy and move to output folder
            shutil.copy(source, os.path.join(destination, str(i+1) + '-' + similar_items[item][i] + '.jpg'))
            print('Number', i+1, ': ' + source)
            if i+1 == args.number:
                break
        print("Files copied into:", destination)
    else:
        print('Item', item, 'not recognized')
    # search = sc.search_catalog(dataset=args.dataset)
    # search.plot_similar()
        
#function to find the item most similar to an image
def visual_search(img):
    search = sc.search_catalog(dataset=args.dataset)
    search.run(img, load_features=True, model=args.transfer_model, data_augmentation=data_augmentation) 
    #print the results
    k=0
    # search.plot_similar()
    print(search.distances, search.NN)
    for i in range(len(search.similar_items)):
        if search.similar_items[i] not in search.similar_items[:i]:#we remove duplicates
            k+=1
            print('Most similar item number', k, ': ' + os.path.join('.', 'data', 'dataset', 'fashion_data', search.similar_items[i] + '.jpg'))
            if k == args.number:
                break

#run the proper function given the --task argument passed to the function   
if args.task == 'fit':
    fit()

elif args.task == 'show_similar_items':
    show_similar_items()
    
elif args.task == 'visual_search':
    visual_search(args.img)