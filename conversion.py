#importing pandas as pd 
import pandas as pd 
import os

path_xslx = "./data"
path_csv = "./csv"
for filename in os.listdir(path_xslx):
        
    # Read and store content 
    # of an excel file  
    read_file = pd.read_excel(path_xslx+"/"+filename) 
    
    # Write the dataframe object 
    # into csv file 
    read_file.to_csv (path_csv+"/"+filename[:-5]+".csv",  
                    index = None, 
                    header=True) 