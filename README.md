# image-similarity
Library built to perform two tasks: (1) find the products in the catalog most similar to a given one based on their images, and (2) from the image taken by a user, find the products in the catalog which look the most similar. The application of (1) is the recommendation of similar products on Websites and applications. The application of (2) is to provide users the possibility to search for products based on images, similarly to https://images.google.com/.

The algorithm used to calculate image similarity is based on transfer learning. In a nutshell, we compute the features of each image using VGG19 and Inception_Resnet_V2 models. We then calculate the similarity of each image given the cosine distance between their feature vectors, and we rank the images with the highest similarity.

For more information, please contact aicanada@decathlon.net

## Getting started
Make sure you have python 3 and the required libraries properly installed (pip install requirements.txt). You can then git clone the project to the desired location
```
git clone https://github.com/decathloncanada/image-classification.git
```
To be able to read files from the preprod.datamining bucket on S3, make sure that you have a config.py file in the root folder of the directory, with the following content:
```
aws_access_key_id={AWS_ACCESS_KEY_ID}
aws_secret_access_key={AWS_SECRET_ACCESS_KEY_ID}
```
where you replace {AWS_ACCESS_KEY_ID} and {AWS_SECRET_ACCESS_KEY_ID} with your own keys.

## Extract the images from the catalog
The images of the models in the catalog can be extracted by running the following command:
```
python main.py --task extract_images --dpt_number {DPT_NUMBER} --domain_id {DOMAIN_ID}
```
where you replace {DPT_NUMBER} with the number of the department (for instance, 371 for ice hockey) and {DOMAIN_ID} with the id of the domain (for instance, 0341 for decathlon.ca). When making this call, the library will load the following file on S3:
```
preprod.datamining/images/data/pixlIDs_domain_id_{DOMAIN_ID}_dpt_num_department_{DPT_NUMBER}000.gz'
```
This file must contain at least two columns, *product_id_model*, the id of the models in the catalog, and *id*, the pixl ids of the images of these models.

The library will then run through all the pixl ids, and download the images from contents.mediadecathlon.com. 

## Find the products most similar to a given one

## Find the products most similar to a given image


## Roadmap
