import pandas as pd
import wget
import os
import requests

clothes_data=pd.read_csv('./data/web_scraping/clothes_data.csv')


for i in range(len(clothes_data)):
    destination = os.path.join('data','dataset','Dresses-' + clothes_data.at[i, 'source'])
    try:
        os.makedirs(destination, exist_ok=True)
    except OSError:
        print ("Creation of the directory %s failed" % destination)
    else:
        print ("Successfully created the directory %s " % destination)

    files = os.listdir(destination)
    if len(files) > 125: continue

    url=clothes_data.at[i, 'image_url']
    filename =  clothes_data.at[i, 'source'] + "-" + str(i+1) + ".jpg"
    if not pd.isna(url):
        r = requests.get(url)
        with open(os.path.join(destination, filename), 'wb') as f:
            f.write(r.content)
        # wget.download(url, os.path.join(destination, filename))
    if i%100==0:
        print(i)

