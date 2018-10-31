# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:21:04 2018

Script to extract and save images based on pixl IDs

@author: AI team
"""
import boto3
import io
import pandas as pd
import shutil
import urllib.request

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

try:
    from config import aws_access_key_id, aws_secret_access_key
except:
    #grab keys from the environment for Jenkins runs
    aws_access_key_id = os.environ.get('aws_access_key_id')
    aws_secret_access_key = os.environ.get('aws_secret_access_key')
    
class extract_images():
    
    def __init__(self):
        self.models = None
        
    #method to extract pixl IDs and model codes from S3
    def _extract_IDs(self, dpt_num_department, domain_id, sep=";",
                     header=0, names=None, compression='gzip'):
        
        session = boto3.session.Session(region_name='EU-west-1')
        s3client = session.client('s3',
                              aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key)
        
        response = s3client.get_object(Bucket='preprod.datamining',
                                   Key='images/data/pixlIDs_domain_id_' + domain_id + '_dpt_num_department_' + str(dpt_num_department) + '000.gz')
        
        self.models = pd.read_csv(io.BytesIO(response['Body'].read()), 
                         sep=sep, header=header,
                         names=names, compression=compression)
        
    #main method: loops through pixl IDs, and extract images 
    def run(self, dpt_num_department=371, domain_id='0341', delete_images=True):
        
        #extract the models/pixl IDs in a dataframe
        self._extract_IDs(dpt_num_department=dpt_num_department,
                          domain_id=domain_id)
        
        folder = parentdir + '\\data\\dataset\\' + 'dpt_num_department_' + str(dpt_num_department)
        #if we delete previous images
        if delete_images:
            print('Deleting previous images')            
            if os.path.exists(folder):
                for the_file in os.listdir(folder):
                    file_path = os.path.join(folder, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path): shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)
                
        
        #create dataset directory if it does not exists
        if not os.path.exists(folder):
            os.makedirs(folder)
               
        #loop through the pixl IDs, and load the images
        k = 0
        for index, row in self.models.iterrows():
            url = 'https://contents.mediadecathlon.com/p' + str(row['id']) + '/'
            file = folder + '\\' + str(row['product_id_model']) + '_' + str(row['id']) + '.jpg'
            #extract and save the image
            try:
                urllib.request.urlretrieve(url, file)
                k += 1
                print('Image retrieved number', k)
            except:
                print('Image not retrieved for model', row['product_id_model'], 'pixlID', row['id'])
    
    
if __name__ == '__main__':
    extracter = extract_images()
    extracter.run(dpt_num_department=371, domain_id='0341')
        