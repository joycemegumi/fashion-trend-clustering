# image-similarity
Library built to perform two tasks: (1) find the products in the catalog most similar to a given one based on their images, and (2) from the image taken by a user, find the products in the catalog which look the most similar. The application of (1) is the recommendation of similar products on Websites and applications. The application of (2) is to provide users the possibility to search for products based on images, similarly to https://images.google.com/.

The algorithm used to calculate image similarity is based on transfer learning. In a nutshell, we compute the features of each image using VGG19 and Inception_Resnet_V2 models. We then calculate the similarity of each image given the cosine distance between their feature vectors, and we rank the images with the highest similarity.

For more information, please contact aicanada@decathlon.net

## Getting started
Make sure you have python 3 and the required libraries properly installed (pip install requirements.txt). You can then git clone the project to the desired location
```
git clone https://github.com/decathloncanada/image-classification.git
```

## Extract the images from the catalog

## Find the products most similar to a given one

## Find the products most similar to a given image


## Roadmap
