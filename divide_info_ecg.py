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
import pickle

# Get PKL
pkl_path = 'D:\\DATA_STORAGE\\EXTRACTED\\PKL\\KUMC\\'

pkl_files = os.listdir(pkl_path)
infile = open(pkl_path + pkl_files[4], 'rb')
df_pkl = pickle.load(infile)
infile.close()
print(pkl_files[4])

#
label_dict = {'Atrial flutter':'AFL',
              'Atrial fibrillation':'AF',
              'Normal sinus rhythm':'NSR',
              'Premature atrial complexes':'APC',
              'Premature ventricular complexes':'VPC',
              'Sinus bradycardia':'SB',
              'Sinus tachycardia':'ST',
              'Wide QRS tachycardia (VT)':'VT',
              'with 1st degree AV block':'1AVB',
              'with 2nd degree AV block (Mobitz I)':'Wench',
              'with 2nd degree AV block (Mobitz II)':'2AVB',
              'with complete heart block (CHB)':'CAVB'}

label = []
i = 0
for i in range(0,len(df_pkl['temp_Label'])):
    label.append(label_dict.get(df_pkl['temp_Label'][i]))
df_pkl['label'] = label

# Split INFO & ECG
df_pkl['filename'] = df_pkl['PersonID']+'_'+df_pkl['Study_Date']+'_'+df_pkl['Study_Time']+'_'+df_pkl['label']

info_list = ['filename','PersonID', 'Study_Date', 'Gender', 'Age', 'HeartRate','Statement',
             'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected', 'PAxis',
             'RAxis', 'TAxis', 'temp_Label']

ecg_list = ['filename','Amplitude', 'wave_I', 'wave_II', 'wave_III', 'wave_aVR', 'wave_aVL', 'wave_aVF',
            'wave_V1', 'wave_V2', 'wave_V3', 'wave_V4', 'wave_V5', 'wave_V6']

df_info = df_pkl[info_list]
df_ecg = df_pkl[ecg_list]

# Export ECG
ecg_path = 'D:\\DATA_STORAGE\\EXTRACTED\\PKL\\'
ecg_name = 'KUMC_01_20210830_ecg.pkl'
with open(ecg_path + ecg_name, 'wb') as handle:
    pickle.dump(df_ecg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(ecg_name)

# Preprocess INFO
df_info['patient_id'] = df_info['PersonID']
df_info['hospital_name'] = 'KUMC'
df_info['method'] = '12lead'
df_info['age'] = df_info['Age']
df_info['hr'] = df_info['HeartRate']
df_info['date'] = df_info['Study_Date']
df_info['gender'] = df_info['Gender']
df_info['month'] = '0'

info_orders = ['filename', 'patient_id', 'hospital_name', 'method',
               'age', 'month', 'gender', 'hr', 'date', 'Statement','temp_Label',
               'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected', 'PAxis',
               'RAxis', 'TAxis'
               ]

df_info_total = df_info[info_orders]


m_list = ['Male','MALE']
f_list = ['Female', 'FEMALE']
u_list = ['Unknown','UNKNOWN']
df_info_total.loc[df_info_total['gender'].isin(m_list), 'gender'] = '1'
df_info_total.loc[df_info_total['gender'].isin(f_list), 'gender'] = '2'
df_info_total.loc[df_info_total['gender'].isin(u_list), 'gender'] = '3'

# Export INFO
df_info_total['filename'] = df_info_total['filename'].str.replace(':','_')
info_path = 'D:\\DATA_STORAGE\\EXTRACTED\\PKL\\KUMC\\'
info_name = 'KUMC_01_20210830_info_v1_1.pkl'
with open(info_path + info_name, 'wb') as handle:
    pickle.dump(df_info_total, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(info_name)

