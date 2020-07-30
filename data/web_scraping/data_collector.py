from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import time
import os
import traceback as tb

from requests import get

hdr = {  'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36',
         'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
         'accept-encoding': 'gzip, deflate, br',
         'accept-language': 'en-US,en;q=0.9'}


def scrapeImgUrl():
    
    if not os.path.exists('./data/web_scraping/clothes_data.csv'):
        clothes_data=pd.DataFrame(columns=[
            'image_url',
            'product_name',
            'source'
        ])
        clothes_data.to_csv('./data/web_scraping/clothes_data.csv')
        print('clothes_data.csv saved')

    clothes = pd.read_csv('./data/web_scraping/clothes.csv')
    print('reading clothes.csv')

    clothes_data = pd.read_csv('./data/web_scraping/clothes_data.csv')


    clothes_rows=[]
    print(len(clothes_data), len(clothes))
    save_every=10

    for i in range(len(clothes_data), len(clothes)):
        # time.sleep(.1)
        print('inside')
        try:
            clothes_URL=clothes.at[i, 'URL']
            currentBrand = clothes.at[i, 'source']
            selector = selectors[currentBrand]
            print(clothes_URL + ' url printing')
            clothes_page=get(clothes_URL, headers=hdr)
            print('printing clothes_page')
            clothes_soup=BeautifulSoup(clothes_page.content, 'html.parser')
            print('Parsing clothes urls')

            item=dict()

            #Save the URL\ of the image of the clothes cover to be downloaded later
            image_url = clothes_soup \
                .find(selector['imgParentEl'], attrs=selector['imgParentAttrs'])
            if selector['imgAttrs'] is None:
                item['image_url'] = image_url.attrs['href']
            elif image_url:
                image_url = image_url.find('img', selector['imgAttrs'])
                item['image_url'] = image_url.attrs['src']
                if 'https' in item['image_url']:
                    item['image_url'] = image_url.attrs['src']
                else:
                   item['image_url'] = 'https:' + image_url.attrs['src']
                print('image urls found')
            else:
                item['image_url']=''
                print('no image url found')
            print(i, item['image_url'])

            #name of the product
            product_name = clothes_soup.find(selector['titleEl'], attrs=selector['titleAttrs'])
            if product_name:
                item['product_name'] = product_name.text.replace('\n','').strip()
                print('product name found')
            else:
                item['product_name']=''
                print('no product name found')
            print(i, item['product_name'])

            if item['product_name'] == '' or item['image_url'] == '': continue
            #source of the product (e.g zara, asos etc)
            item['source'] = currentBrand

            clothes_rows.append(item)

            if i % save_every == 0:
                clothes_data.append(pd.DataFrame.from_dict(clothes_rows)).to_csv('./data/web_scraping/clothes_data.csv', mode='a', header=False, index=False, )
                clothes_rows = []
                print('clothes_row', i, 'saved as csv')
        except Exception as e:
            print('Skipping url as no image was found')


selectors = {
    'asos': {
        'imgParentEl': 'div',
        'imgParentAttrs': {'id':'product-gallery'},
        'imgAttrs': {},
        'titleEl': 'h1',
        'titleAttrs': {}
    },
    'zara': {
        'imgParentEl': 'a',
        'imgParentAttrs': {'class':'main-image'},
        'imgAttrs': None,
        'titleEl': 'h1',
        'titleAttrs': {'class': 'product-name'}
    },
    'topshop': {
        'imgParentEl': 'div',
        'imgParentAttrs': {'class':'Carousel-images'},
        'imgAttrs': {'class':'Carousel-image'},
        'titleEl': 'h1',
        'titleAttrs': {}
    },
    'h&m': {
        'imgParentEl': 'div',
        'imgParentAttrs': {'class':'product-detail-main-image-container'},
        'imgAttrs': {},
        'titleEl': 'h1',
        'titleAttrs': {}
    },
    
}

scrapeImgUrl()