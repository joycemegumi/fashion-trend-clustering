import os
#Defining a function to rename our images by brand and nummber
def main(): 
    #Store name is changed each time
    store = 'TopShop'
    wd = os.path.join('./data', 'dataset', 'Dresses_' + store)
    print(wd)
    for count, filename in enumerate(os.listdir(wd)):
        print(filename)
        # If the filename includes a duplicate (1) then skip it and continue to the next filename
        if '(1)' in filename:
            continue
        dst = os.path.join('data', 'dataset','fashion_data', store + "-" + str(count+1) + ".jpg")
        src = os.path.join(wd, filename)

        # rename() function will rename all the files 
        os.rename(src, dst) 
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main()