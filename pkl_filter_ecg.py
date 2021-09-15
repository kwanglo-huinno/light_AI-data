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
save_path = 'D:\\DATA_STORAGE\\EXTRACTED\\PKL\\KUMC\\ECG_12lead_noAmp_filtered\\'
ecg_path = 'D:\\DATA_STORAGE\\EXTRACTED\\PKL\\KUMC\\ECG_12lead_noAmp\\'
# Read ECG files
ecg_files = os.listdir(ecg_path)
df_ecg = pd.DataFrame()
l = 0
for l in range(0,len(ecg_files)):
    infile = open(ecg_path + ecg_files[l], 'rb')
    df_ecg = pickle.load(infile)
    infile.close()
    print(ecg_files[l])

    # Filter ECG
    df_ecg_info = df_ecg.iloc[:,:2]
    df_ecg_raw = df_ecg.iloc[:,2:]

    df_ecg_temp = df_ecg_raw.astype(np.float32)

    # Scipy
    # bandpass
    print('Applying Bandpass 0.5-100hz')
    ecg_bandpass = []
    for i in range(0,len(df_ecg_temp)):
        ecg_array = np.array(df_ecg_temp.loc[i])
        b,a = signal.butter(2, [0.5,100], btype='bandpass',fs=250)
        filter_bp = signal.filtfilt(b, a, ecg_array)
        ecg_bandpass.append(filter_bp)
    df_bandpass = pd.DataFrame(ecg_bandpass)

    # 60hz notch
    print('Applying Notch 60hz')
    ecg_notch= []
    for i in range(0,len(df_bandpass)):
        ecg_array = np.array(df_bandpass.loc[i])
        b, a = signal.iirnotch(60, 1, 250)  # (target_hz, quality factor, Hz)
        df_filter_n = signal.lfilter(b, a, ecg_array)
        ecg_notch.append(df_filter_n)
    df_scipy = pd.DataFrame(ecg_notch)

    df_ecg_filtered = pd.concat([df_ecg_info,df_scipy],axis=1)

    save_name = ecg_files[l][:-4]+'_filtered.pkl'

    with open(save_path + save_name, 'wb') as handle:
        pickle.dump(df_ecg_filtered, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Test plot

from matplotlib import pyplot as plt

q = 5
plt.figure(figsize=(20, 4))
plt.plot(df_ecg_raw.loc[q])
plt.plot(df_scipy.loc[q])
plt.title(df_ecg_info['filename'][q])
plt.show()
