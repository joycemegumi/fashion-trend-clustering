# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:36:37 2018

Main function to find most similar items to a given item or image

@author: AI team
"""

import argparse
import os
import sys
import inspect
import pickle

from random import randint
from src import find_similar as fs
from src import search_catalog as sc

ROOTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
DATA_TRAINING_PATH = '/data/trained_models/'

# if args.transfer_model not in ['VGG', 'Inception_Resnet']:
#     print('transfer_model not supported')
#     args.task = 'pass'

def fit(arguments):
    """
    Function to build the dictionary of most similar items
    """
    sim = fs.find_similar(dataset=arguments.dataset)
    sim.fit(k=arguments.number, model=arguments.transfer_model,
            data_augmentation=arguments.data_augmentation, save_similar_items=True)


def show_similar_items(arguments):
    """
    Function to show the most similar items to a given item_id
    """
    #load the similar items dictionary
    try:
        with open(ROOTDIR + DATA_TRAINING_PATH + str(arguments.dataset) + '_model_' + arguments.transfer_model + '.pickle', 'rb') as pickle_file:
            similar_items = pickle.load(pickle_file)
    except:
        print("Error {0} Not exist.".format(ROOTDIR + DATA_TRAINING_PATH + str(arguments.dataset) + '_model_' + arguments.transfer_model + '.pickle'))
        sys.exit(1)
    #if no item id is provided, randomly select one
    all_items = [i for i in similar_items]
    if arguments.item is None:
        item = all_items[randint(0, len(all_items))]
    else:
        item = arguments.item

    #verify that the item is in the dictionary
    if item in all_items:
        print('Most Similar items to item: {0}'.format(item))
        for i in range(len(similar_items[item])):
            print('Number{0}: {1}'.format(i+1, similar_items[item][i]))
            if i+1 == arguments.number:
                break

    else:
        print('Item', item, 'not recognized')


def visual_search(arguments):
    """
    Function to find the item most similar to an image
    """
    search = sc.search_catalog(dataset=arguments.dataset)
    search.run(arguments.img, \
        load_features=True,
        model=arguments.transfer_model,
        data_augmentation=arguments.data_augmentation)
    k = 0
    for i in range(len(search.similar_items)):
        if search.similar_items[i] not in search.similar_items[:i]:#we remove duplicates
            k += 1
            print('Most similar item number', k, ': ' + str(search.similar_items[i]))
            if k == arguments.number:
                break

def get_option():
    """
    Get option in command line.
    :return: parser
    """
    parser = argparse.ArgumentParser(description='Build a dictionary of most similar items, \
        and search for the most similar item in a dataset')
    parser.add_argument('--task', type=str, choices=["fit", "show_similar_items", "visual_search"],
                        help="""
                        task to perform:
                        fit --> compute features for all the images in the dataset
                        show_similar_items --> show the items most similar to a given item_id
                        visual_search --> find the items most similar to a given image
                        """)
    parser.add_argument('--data_augmentation', action="store_true",
                        help="""
                        If we want (1) or not (0) to consider data augmentation (lr/ud flipping and 90,180,270 degree rotations)
                        """)
    parser.add_argument('--dataset', type=str, default='/data/dataset/',
                        help="""
                        Name of the dataset containing the image of all the items
                        """)
    parser.add_argument('--item', type=str,
                        help="""
                        ID of the item for which we want to show the most similar items in the dataset
                        """)
    parser.add_argument('--img', type=str,
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

    return parser.parse_args()

def main():
    """
    main function.
    """
    arguments = get_option()
    if arguments.task == 'fit':
        fit(arguments)
    elif arguments.task == 'show_similar_items':
        show_similar_items(arguments)
    elif arguments.task == 'visual_search':
        visual_search(arguments)

if __name__ == '__main__':
    main()
