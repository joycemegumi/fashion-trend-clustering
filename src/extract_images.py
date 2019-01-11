# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:21:04 2018

Script to extract and save images based on pixl IDs

@author: AI team
"""
import shutil
from urllib.request import urlopen

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import utils   
   
class extract_images():
    
    def __init__(self):
        self.models = None
        
    #method to extract pixl IDs and model codes from S3
    def _extract_IDs(self, dpt_num_department, domain_id='0341'):        
        self.models = utils.read_S3(dpt_num_department=dpt_num_department,
                                    domain_id=domain_id)
        
    #main method: loops through pixl IDs, and extract images 
    def run(self, dpt_num_department=371, domain_id='0341', delete_images=False,
            target_size=None):
        
        #extract the models/pixl IDs in a dataframe
        self._extract_IDs(dpt_num_department=dpt_num_department,
                          domain_id=domain_id)
        
        folder = parentdir + '\\data\\dataset\\' + 'dpt_num_department_' + str(dpt_num_department) + 'domain_id_' + str(domain_id)
        
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
            
        #get new images, that we had not previously downloaded
        self.old_models = [i[:-4] for i in os.listdir(folder)]
               
        #loop through the pixl IDs, and load the images
        k = 0
        for index, row in self.models.iterrows():
            #see if image already downloaded
            if str(row['product_id_model']) + '_' + str(row['id']) in self.old_models:
                k += 1
            else:
                if target_size is not None:
                    url = 'https://contents.mediadecathlon.com/p' + str(row['id']) + '/sq/' + str(row['id']) + '.jpg?f=' + str(target_size[0]) + 'x' + str(target_size[1])
                else:
                    url = 'https://contents.mediadecathlon.com/p' + str(row['id']) + '/'
                file = folder + '\\' + str(row['product_id_model']) + '_' + str(row['id']) + '.jpg'
                #extract and save the image
                try:
                    request = urlopen(url, timeout=10).read()
                    with open(file, 'wb') as f:
                        f.write(request)
                    k += 1
                    print('Image retrieved number', k)
                except:
                    print('Image not retrieved for model', row['product_id_model'], 'pixlID', row['id'])
                    
                    
if __name__ == '__main__':
    extracter = extract_images()
    extracter.run(dpt_num_department=0, domain_id='0341',
                  target_size=(224, 224))
        