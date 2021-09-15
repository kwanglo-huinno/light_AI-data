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

# Get path
pkl_path = 'D:\\DATA_STORAGE\\EXTRACTED\\PKL\\KUMC\\'
save_path = 'D:\\DATA_STORAGE\\EXTRACTED\\PKL\\KUMC\\ECG_12lead_noAmp\\'

# Read info & ECG files
pkl_files = os.listdir(pkl_path)

# ECG
infile = open(pkl_path + pkl_files[5], 'rb')
df_ecg = pickle.load(infile)
infile.close()
print(pkl_files[5])
'''
['filename', 'Amplitude', 'wave_I', 'wave_II', 'wave_III', 'wave_aVR',
'wave_aVL', 'wave_aVF', 'wave_V1', 'wave_V2', 'wave_V3', 'wave_V4',
'wave_V5', 'wave_V6']
'''
lead_dict = {'wave_I':'lead1',
             'wave_II':'lead2',
             'wave_III':'lead3',
             'wave_aVR':'lead4',
             'wave_aVL':'lead5',
             'wave_aVF':'lead6',
             'wave_V1':'lead7',
             'wave_V2':'lead8',
             'wave_V3':'lead9',
             'wave_V4':'lead10',
             'wave_V5':'lead11',
             'wave_V6':'lead12'}
for l in lead_dict.keys():
    lead_ = l
    lead_num = lead_dict.get(l)
    print(lead_)
    print(lead_num)
    lead_list = ['filename','Amplitude',lead_]
    df_lead = df_ecg[lead_list]
    df_lead['lead'] = lead_num
    df_temp = pd.DataFrame(df_lead[lead_].str.split(',',expand=True))
    ecg_array = np.array(df_temp, dtype=np.float32)
    # ecg_array = ecg_array * df_lead['Amplitude'][0]
    df_ecg_temp = pd.DataFrame(ecg_array)
    # Test Plot
    '''
    from matplotlib import pyplot as plt
    q = 3
    plt.figure(figsize=(20, 4))
    plt.plot(df_ecg_temp.loc[q])
    plt.show()
    '''

    df_lead = pd.concat([df_lead,df_ecg_temp],axis=1)
    df_lead = df_lead.drop(columns=['Amplitude',lead_])
    # filename tnwjd
    df_lead['filename'] = df_lead['filename'].str.replace(':','_')

    save_name = pkl_files[4][:-4]+'_noAmp_'+df_lead['lead'][0]+'.pkl'

    with open(save_path + save_name, 'wb') as handle:
        pickle.dump(df_lead, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(save_name)