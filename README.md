# image-similarity
Library built to perform two tasks: (1) find the items in a dataset most similar to a given one based on their images, and (2) from the image taken by a user, find the items in the dataset which look the most similar. The application of (1) is to build a recommendation system based on item similarity. The application of (2) is to build a visual search engine (similarity to https://images.google.com/), to give users the ability to search for items based on images.

The algorithm used to calculate image similarity is based on transfer learning. In a nutshell, we compute the features of each image using VGG19 and Inception_Resnet_V2 models. We then calculate the similarity of each image given the cosine distance between their feature vectors, and we rank the images with the highest similarity.

For more information, please read [our blog post](https://medium.com/decathlondevelopers/building-a-visual-search-algorithm-in-a-few-steps-using-transfer-learning-ea00cb9fe49c) or contact aicanada@decathlon.net

## Getting started
Make sure you have python 3 and the required libraries properly installed (to install the libraries, run the command pip install -r requirements.txt from, for instance, an Anaconda Prompt). You can then git clone the project to the desired location:
```
git clone https://github.com/decathloncanada/image-similarity.git
```

## Dataset format
Your dataset of images should be be placed in the data/dataset directory. The dataset should respect the following convention:
- the name of each image in the dataset should be unique;
- an underscore is used to indicate an item associated to more than a single image

For instance, the let's say we have the dataset "hockey_products", containing the following images:
```
data/
  dataset/
    hockey_products/
      helmet1.jpg
      helmet2.jpg
      stick1_1.jpg
      stick1_2.jpg
      hockeybag.jpg
```
The library will in this case consider that the dataset is composed of four items (helmet1, helmet2, stick1 and hockeybag), and that the item stick1 is associated to two images (stick1_1.jpg and stick1_2.jpg).


## Services
### Calculation of the dictionary of most similar items 
To find, for each item, the list of the other items in the dataset which are most similar, run the following call:
```
python main.py --task fit --dataset {DATASET} --transfer_model {TRANSFER_MODEL} --number {NUMBER} --data_augmentation {DATA_AUGMENTATION}
```
This will loop through the images found in the dataset of the name given as the --dataset argument. This dataset should be found in the data/dataset directory, and be formated as detailed in the previous sections. For each image, their feature vectors will be calculated and stored in a sqlite database located in the data/database directory. The name of the database is features.db, and the results are stored in a table named features_{DATASET}. Note that the computation of the features can take hours if the dataset contains many (> 1000) images. However, the features can be calculated over multiple sessions as, at each call, only the features not already found in the database are calculated.

Data augmentation can be enabled by passing 1 as the --data_augmentation argument - in this case, the features will be calculated for the original image, as well as the image fliped left-right and up-down, and the image rotated by angles of 90, 180 and 270 degrees. 

After the features have been calculated, the library will loop through all the items, and find the items in the rest of the dataset which look the most similar. The similarity is based on the cosine distance between the feature vectors calculated using the model provided as the --transfer_model argument (two models are currently supported, VGG19 (VGG) and Inception_Resnet_V2 (Inception_Resnet)). The results are saved in a a file named {DATASET}\_model_\{TRANSFER_MODEL}.pickle, found in the data/trained_models directory. This file contains a dictionary {item: [list of most similar items]}. This dictionray provides, for each item, the list of items which are the most similar. The lenght of this list is of 10 by default, but can be controlled using the --number argument.

## Show the items most similar to a given one
Once the dictionary of most similar items has been built (see the previous section), we can run the following call to print the most similar items to a given one.
```
python main.py --task show_similar_items --item {ITEM} --dataset {DATASET} --transfer_model {TRANSFER_MODEL} --number {NUMBER}
```
When making this call, the library first opens the dictionary of most similar items. If the argument --item is not provided, one item is picked randomly, and the most similar items are returned. 

For example, let's assume that you have built the dictionary of most similar items (as described in the following section) for the hockey_products dataset (see the *Dataset format* section). To get the most similar items to the item helmet1, based on the features calculated by Inception_Resnet model, you could run the following call:
```
python main.py --task show_similar_items --item helmet1 --dataset hockey_products --transfer_model Inception_Resnet
```
The response would look like:
```
Most similar items to item: helmet1
Number 1 : helmet2
Number 2 : hockeybag
Number 3 : stick1
```


## Perform a visual search of the dataset
Once the features for all the images have been calculated (see the section *Calculation of the dictionary of most similar items*), the items most similar to a given image can be found by providing the path to the image:
```
python main.py --task visual_search --img {IMG} --dataset {DATASET} --transfer_model {TRANSFER_MODEL} --data_augmentation {DATA_AUGMENTATION}
```
where {IMG} is the path to the given image.

For instance, let's say you have the image of a hockey stick, located at path ./test/images/stick.jpg. You want to find the three items in the hockey_products dataset which look the most similar, based on the feature vectors calculated using Inception_Resnet model and considering the images obtained by data augmentation. You can then run the following call:
```
python main.py --task visual_search --img ./test/images/stick.jpg --dataset hockey_products --transfer_model Inception_Resnet --data_augmentation 1 --number 3
```
The response could look like:
```
Most similar item number 1 : stick1
Most similar item number 2 : helmet2
Most similar item number 3 : helmet1
```

## Roadmap
Future improvements of the library in the work includes enabling the removal of the background color in images taken by users, and improving the ensemble approach used in the presence of multiple images by item in the dataset.
