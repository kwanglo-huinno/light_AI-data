import numpy as np
import pandas as pd
import os
from shutil import copyfile
import timeit
import pickle

pkl_path = 'D:\\DATA_STORAGE\\EXTRACTED\\PKL\\'

pkl_files = os.listdir(pkl_path)
infile = open(pkl_path + pkl_files[0], 'rb')
df_pkl = pickle.load(infile)
infile.close()

folder_list = list(df_pkl['temp_Label'].unique())
folder_path = 'D:\\DATA_STORAGE\\EXTRACTED\\IMG\\KUMC\\'
for f in folder_list:
    try:
        os.mkdir(folder_path+f)
    except OSError:
        pass
    else:
        pass