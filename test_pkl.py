import numpy as np
import pandas as pd
import os
from shutil import copyfile
import timeit
import pickle
import cv2
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn import preprocessing
from scipy import signal
from scipy.signal import butter, lfilter
import copy
import timeit
from tqdm import tqdm

# Get path
pkl_path = 'D:\\DATA_STORAGE\\EXTRACTED\\PKL\\KUMC\\'
# Read info & ECG files
pkl_files = os.listdir(pkl_path)
# INFO
p = 6
infile = open(pkl_path + pkl_files[p], 'rb')
df_info = pickle.load(infile)
infile.close()
print(pkl_files[p])

df_info['filename'] = df_info['filename'].str.replace(':','_')

with open(pkl_path + pkl_files[p], 'wb') as handle:
    pickle.dump(df_info, handle, protocol=pickle.HIGHEST_PROTOCOL)