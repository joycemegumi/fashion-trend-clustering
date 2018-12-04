# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:25:06 2018

Helper functions

@author: AI team
"""
import boto3
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

try:
    from config import aws_access_key_id, aws_secret_access_key
except:
    #grab keys from the environment for Jenkins runs
    aws_access_key_id = os.environ.get('aws_access_key_id')
    aws_secret_access_key = os.environ.get('aws_secret_access_key')

#function to extract data from S3
def read_S3(dpt_num_department, domain_id, sep=";",
            header=0, names=None, compression='gzip'):
        
        session = boto3.session.Session(region_name='EU-west-1')
        s3client = session.client('s3',
                              aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key)
        
        response = s3client.get_object(Bucket='preprod.datamining',
                                   Key='images/data/pixlIDs_domain_id_' + domain_id + '_dpt_num_department_' + str(dpt_num_department) + '000.gz')
        
        models = pd.read_csv(io.BytesIO(response['Body'].read()), 
                         sep=sep, header=header,
                         names=names, compression=compression)
        
        return models

#function to plot models similar to a given model or a given image
def plot_similar(path_to_img, path_to_similar_mdls, img_name='', subplots=(3,3)):
       
    # Create figure with sub-plots.
    fig, axes = plt.subplots(subplots[0], subplots[1])

    # Adjust vertical spacing.
    hspace = 0.3
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    interpolation = 'spline16'
    
    for i, ax in enumerate(axes.flat):
        # There may be less models than slots in the plot - make sure it doesn't crash.
        if i <= len(path_to_similar_mdls):
            # we keep the first image to show the image we are investigating
            if i==0:
                path = path_to_img
            # the other images are those of the similar models
            else:
                path = path_to_similar_mdls[i-1]

            image = np.asarray(plt.imread(path))
            
            ax.imshow(image,
                      interpolation=interpolation)

            # set the xlabel
            if i==0:
                if img_name is not '':
                    xlabel = 'Model/picture we are investigating:' + img_name
                else:
                    xlabel = 'Model/picture we are investigating'
            else:
                trimmed_name = path.split('_')[-2]
                model_ID = trimmed_name.split('/')[-1]
                
                xlabel = "Similar model {}: {}".format(i, model_ID)
                
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()