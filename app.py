# -*- coding: utf-8 -*-
"""
Router for the endpoint of the image similarity library

@author: AI team
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

from tensorflow.python.keras import models
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import Model

from src import find_similar as fs
from src import search_catalog as sc

app = Flask(__name__)
CORS(app)

#function to load VGG19 model
model = None
def load_model():
    global classes
    global graph
    global model
    print("Loading VGG19 pre-trained model...")
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output) 
    
@app.route('/image', methods=['POST'])
def image():
    #initialize the returned data dictionary
    data = {"success": False}
    
    try:
        search = sc.search_catalog(dpt_num_department=request.args.get('dpt_number'))
        search.model = model
        search.run(request.args.get('path'), load_features=True, dataset_augmentation=True, load_model=False)
        data['similar_models'] = search.similar_models
            
        # indicate that the request was a success
        data["success"] = True
    except:
        pass

    # return the data as a JSON
    return jsonify(data)
    
@app.route('/model', methods=['GET'])
def model():
    #initialize the returned data dictionary
    data = {"success": False}
    
    try:
        path = 'data\\trained_models\\'
        with open(path + 'similar_models_dpt_num_department_' + str(request.args.get('dpt_number')) + '.pickle', 'rb') as file:
            similar_models = pickle.load(file)
         
        print(str(request.args.get('ID')))
        data['similar_models']: similar_models[str(request.args.get('ID'))]
            
        # indicate that the request was a success
        data["success"] = True
    except:
        pass

    # return the data as a JSON
    return jsonify(data)

if __name__ == '__main__':
    print('Loading classification model')
    load_model()
    app.run()