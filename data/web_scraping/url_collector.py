from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import time
import os
from requests import get

#Sometimes not including the header results in a failed response
hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
         'Referer': 'https://cssspritegenerator.com',
         'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
         'Accept-Encoding': 'none',
         'Accept-Language': 'en-US,en;q=0.8'}


if not os.path.exists('./data/web_scraping/clothes.csv'):
    clothes=pd.DataFrame(columns=[
        'URL',
        'source'
    ])
    clothes.to_csv('./data/web_scraping/clothes.csv')
    print('clothes.csv saved')

def scrapeUrl(currentBrand):
    selector = selectors[currentBrand]

    LIST_URL = selector['URL']

    clothes = {'URL' : [], 'source' : []}

    #no. of pages this list has
    num_pages = selector['pages']

    for i in range(1, num_pages+1):
        print(f'Reading page {i}')
        list_page_url = f'{LIST_URL}&page={i}'
        list_page = get(list_page_url, headers=hdr)
        list_soup = BeautifulSoup(list_page.content, 'html.parser')
        #Add our sources for each item link
        clothes_table = list_soup.find(selector['tableParentEl'], attrs=selector['tableParentAttrs'])
        rows = clothes_table.find_all(selector['rowsEl'], attrs=selector['rowsAttrs'] )
        for r in rows:
            row = r.find(selector['rowsCEl'], attrs=selector['rowsCAttrs'])
            if row is None: continue
            clothes_url = row.attrs[selector['rowsC2Attrs']]
            if 'https' in clothes_url:
                clothes['URL'].append(row.attrs[selector['rowsC2Attrs']])
            else:
                clothes['URL'].append(selector['domain'] + row.attrs[selector['rowsC2Attrs']])
            clothes['source'].append(currentBrand)
            
        print(clothes)
    clothes_df = pd.DataFrame.from_dict(clothes)
    print(clothes_df['URL'][0])
    clothes_df.to_csv('./data/web_scraping/clothes.csv', mode='a', header=False)


selectors = {
    'zara': {
        'URL': "https://www.zara.com/us/en/woman-new-in-l1180.html?v1=1549286",
        'pages': 1,
        'tableParentEl': 'ul',
        'tableParentAttrs': {'class': '_productList'},
        'rowsEl': 'li',
        'rowsAttrs': {'class': '_product'},
        'rowsCEl': 'a',
        'rowsCAttrs': {'class':'_item'},
        'rowsC2Attrs': 'href',
        'domain': ''
    },
    'asos': {
        'URL': "https://www.asos.com/us/women/new-in/new-in-clothing/cat/?cid=2623&currentpricerange=5-950&nlid=ww|new%20in|new%20products&refine=attribute_1047:8416",
        'pages': 4,
        'tableParentEl': 'div',
        'tableParentAttrs': {'data-auto-id': 'productList'},
        'rowsEl': 'article',
        'rowsAttrs': {},
        'rowsCEl': 'a',
        'rowsCAttrs': {},
        'rowsC2Attrs': 'href',
        'domain': ''
    },
    'topshop': {
        'URL': "https://us.topshop.com/en/tsus/category/clothing-70483/dresses-70497/N-b1lZdgm?Nrpp=24&Ns=product.freshnessRank%7C0&siteId=%2F13052",
        'pages': 6,
        'tableParentEl': 'div',
        'tableParentAttrs': {'class': 'ProductList-products'},
        'rowsEl': 'div',
        'rowsAttrs': {'class': 'Product-images-container'},
        'rowsCEl': 'a',
        'rowsCAttrs': {},
        'rowsC2Attrs': 'href',
        'domain': 'https://us.topshop.com'
    },
    'h&m': {
        'URL': "https://www2.hm.com/en_us/women/products/dresses.html?product-type=ladies_dresses&sort=newProduct&image-size=small&image=model&offset=0&page-size=36",
        'pages': 4,
        'tableParentEl': 'ul',
        'tableParentAttrs': {'class': 'small'},
        'rowsEl': 'li',
        'rowsAttrs': {'class': 'product-item'},
        'rowsCEl': 'a',
        'rowsCAttrs': {},
        'rowsC2Attrs': 'href',
        'domain': 'https://www2.hm.com'
    }
}

for selector in selectors:
    scrapeUrl(selector)
