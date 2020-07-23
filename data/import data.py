import os

def main(): 
    store = 'ZARA'
    # wd = './Users/joycemegumi/Documents/GitHub/image-similarity/data/dataset/Dresses_ASOS 
    wd = os.path.join('./data', 'dataset', 'Dresses_' + store)
    #print(os.listdir(wd))
    print(wd)
    for count, filename in enumerate(os.listdir(wd)):
        print(filename)
        if '(1)' in filename:
            continue
        dst = os.path.join('data', 'dataset','fashion_data', store + "-" + str(count+1) + ".jpg")
        src = os.path.join(wd, filename)
        # dst ='Dresses_ASOS'+ dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main()