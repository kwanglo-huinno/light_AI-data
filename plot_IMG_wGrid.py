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
ecg_path = 'D:\\DATA_STORAGE\\EXTRACTED\\PKL\\KUMC\\ECG_12lead_noAmp_filtered\\'
save_path = 'D:\\DATA_STORAGE\\EXTRACTED\\IMG\\KUMC\\TEST\\4stack_noAmp_test\\'
# Read info & ECG files
pkl_files = os.listdir(pkl_path)
# INFO
p = 6
infile = open(pkl_path + pkl_files[p], 'rb')
df_info = pickle.load(infile)
infile.close()
print(pkl_files[p])

#target_label = 'Atrial fibrillation'
#df_info = df_info[df_info['temp_Label']==target_label].reset_index(drop=True)
# ECG
ecg_files = os.listdir(ecg_path)
df_ecg = pd.DataFrame()
l = 0
for l in range(0,len(ecg_files)):
    infile = open(ecg_path + ecg_files[l], 'rb')
    df_ecg_temp = pickle.load(infile)
    infile.close()
    print(ecg_files[l])

    #df_ecg_temp = df_ecg_temp[df_ecg_temp['filename'].isin(df_info['filename'])].reset_index(drop=True)

    df_ecg = df_ecg.append(df_ecg_temp).reset_index(drop=True)
print('Done reading ECG')
# Test plot
'''
from matplotlib import pyplot as plt

q = 5
plt.figure(figsize=(20, 4))
plt.plot(df_ecg_raw.loc[q])
plt.plot(df_scipy.loc[q])
plt.title(df_ecg_info['filename'][q])
plt.show()
'''

############################### Plot IMG #######################################
# Reference voltage 추가
arr_zero = np.zeros(50)
arr_one = np.ones(100) * 200
arr_refer = pd.DataFrame(list(np.hstack((arr_zero,arr_one,arr_zero))))

# Create loop here
# Plot IMGs
filename = list(df_ecg['filename'].unique())
f = 0
for f in tqdm(range(0,10)): #len(filename)
    # For each filename
    df_target_ecg = df_ecg[df_ecg['filename'] == filename[f]].reset_index(drop=True)
    df_target_info = df_info[df_info['filename'] == filename[f]].reset_index(drop=True)
    # Set target row to plot
    lead1 = df_target_ecg[df_target_ecg['lead'] == 'lead1'].drop(columns=['filename', 'lead']).reset_index(drop=True)
    lead2 = df_target_ecg[df_target_ecg['lead'] == 'lead2'].drop(columns=['filename', 'lead']).reset_index(drop=True)
    lead3 = df_target_ecg[df_target_ecg['lead'] == 'lead3'].drop(columns=['filename', 'lead']).reset_index(drop=True)
    lead4 = df_target_ecg[df_target_ecg['lead'] == 'lead4'].drop(columns=['filename', 'lead']).reset_index(drop=True)
    lead5 = df_target_ecg[df_target_ecg['lead'] == 'lead5'].drop(columns=['filename', 'lead']).reset_index(drop=True)
    lead6 = df_target_ecg[df_target_ecg['lead'] == 'lead6'].drop(columns=['filename', 'lead']).reset_index(drop=True)
    lead7 = df_target_ecg[df_target_ecg['lead'] == 'lead7'].drop(columns=['filename', 'lead']).reset_index(drop=True)
    lead8 = df_target_ecg[df_target_ecg['lead'] == 'lead8'].drop(columns=['filename', 'lead']).reset_index(drop=True)
    lead9 = df_target_ecg[df_target_ecg['lead'] == 'lead9'].drop(columns=['filename', 'lead']).reset_index(drop=True)
    lead10 = df_target_ecg[df_target_ecg['lead'] == 'lead10'].drop(columns=['filename', 'lead']).reset_index(drop=True)
    lead11 = df_target_ecg[df_target_ecg['lead'] == 'lead11'].drop(columns=['filename', 'lead']).reset_index(drop=True)
    lead12 = df_target_ecg[df_target_ecg['lead'] == 'lead12'].drop(columns=['filename', 'lead']).reset_index(drop=True)

    # Append reference voltage
    lead1 = lead1.T.append(arr_refer).reset_index(drop=True).T
    lead2 = lead2.T.append(arr_refer).reset_index(drop=True).T
    lead3 = lead3.T.append(arr_refer).reset_index(drop=True).T
    lead4 = lead4.T.append(arr_refer).reset_index(drop=True).T
    lead5 = lead5.T.append(arr_refer).reset_index(drop=True).T
    lead6 = lead6.T.append(arr_refer).reset_index(drop=True).T
    lead7 = lead7.T.append(arr_refer).reset_index(drop=True).T
    lead8 = lead8.T.append(arr_refer).reset_index(drop=True).T
    lead9 = lead9.T.append(arr_refer).reset_index(drop=True).T
    lead10 = lead10.T.append(arr_refer).reset_index(drop=True).T
    lead11 = lead11.T.append(arr_refer).reset_index(drop=True).T
    lead12 = lead12.T.append(arr_refer).reset_index(drop=True).T

    # Set features
    # np.zeros(y,x,3 color channel)
    img = np.zeros((1260, 2180, 3), np.uint8)
    img.fill(255)
    # 여기서 색상인 (255,0,0) 에 대해서는 RGB가 아닌 BGR 순서임.

    # Set features
    margin = 50
    gridSize = 40
    offset = 200
    x_offset = 50
    y_offset = 100
    textSpace = 200
    img_col = 2180
    img_row = 1260

    # Set grid
    for y_pos in range(0, 25):
        # start(x,y) / end(x,y) / color / px
        img = cv2.line(img, (margin, textSpace + margin + gridSize * y_pos),
                       (img_col - margin, textSpace + margin + gridSize * y_pos), (100, 100, 255),
                       1)  # (start,ende,color,px)
    for x_pos in range(0, 53):
        img = cv2.line(img, (margin + gridSize * x_pos, textSpace + margin),
                       (margin + gridSize * x_pos, img_row - margin), (100, 100, 255), 1)  # (start,ende,color,px)

    # Set font features
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 0.8
    fontColor = (0, 0, 0)
    lineType = 2
    # Set writing position
    x_lead_std = int(margin + 10)
    y_lead_std = int(textSpace + margin + 30)
    x_lead_augmented = int(margin + 10 + (img_col - margin * 2 - 80) / 4)

    # Write lead index
    img = cv2.putText(img, 'I', (x_lead_std, y_lead_std), font, fontScale, fontColor, lineType)
    img = cv2.putText(img, 'II', (x_lead_std, y_lead_std + gridSize * 6), font, fontScale, fontColor, lineType)
    img = cv2.putText(img, 'III', (x_lead_std, y_lead_std + gridSize * 12), font, fontScale, fontColor, lineType)
    img = cv2.putText(img, 'II', (x_lead_std, y_lead_std + gridSize * 18), font, fontScale, fontColor, lineType)
    img = cv2.putText(img, 'aVR', (x_lead_augmented, y_lead_std), font, fontScale, fontColor, lineType)
    img = cv2.putText(img, 'aVL', (x_lead_augmented, y_lead_std + gridSize * 6), font, fontScale, fontColor, lineType)
    img = cv2.putText(img, 'aVF', (x_lead_augmented, y_lead_std + gridSize * 12), font, fontScale, fontColor, lineType)
    img = cv2.putText(img, 'V1', (x_lead_augmented * 2, y_lead_std), font, fontScale, fontColor, lineType)
    img = cv2.putText(img, 'V2', (x_lead_augmented * 2, y_lead_std + gridSize * 6), font, fontScale, fontColor,
                      lineType)
    img = cv2.putText(img, 'V3', (x_lead_augmented * 2, y_lead_std + gridSize * 12), font, fontScale, fontColor,
                      lineType)
    img = cv2.putText(img, 'V4', (x_lead_augmented * 3, y_lead_std), font, fontScale, fontColor, lineType)
    img = cv2.putText(img, 'V5', (x_lead_augmented * 3, y_lead_std + gridSize * 6), font, fontScale, fontColor,
                      lineType)
    img = cv2.putText(img, 'V6', (x_lead_augmented * 3, y_lead_std + gridSize * 12), font, fontScale, fontColor,
                      lineType)

    # Write info
    info_names = ['filename', 'patient_id', 'age', 'gender', 'hr', 'date']
    info_list = list(df_target_info[info_names].loc[0])
    for info in range(0, len(info_list)):
        img = cv2.putText(img, info_names[info]+': '+str(info_list[info]), (margin + 10, 35 + info * 20), font, 0.6, fontColor, lineType)
    # Write statement
    statement_list = df_target_info['Statement'][0].split(',')
    for state in range(0, len(statement_list)):
        img = cv2.putText(img, statement_list[state],
                          (int(margin + (img_col - margin * 2 - 80) / 4 * 2), 35 + state * 20), font, 0.6, fontColor,
                          lineType)

    # Data
    thickness = 2
    partial_lead = int(5000 / 4)
    # Lead 1 - I
    baseY = margin + gridSize * 3
    startX = margin
    for i in range(1, partial_lead):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5),
                        int(textSpace + baseY - (gridSize * 2 * lead1.loc[0].get(i) / 200))),
                       (int(startX + i / 2.5),
                        int(textSpace + baseY - (gridSize * 2 * lead1.loc[0].get(i+1) / 200))),
                       (0, 0, 0), thickness)

    # Lead 2 - II
    baseY = margin + gridSize * 9
    startX = margin
    for i in range(1, partial_lead):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5),
                        int(textSpace + baseY - (gridSize * 2 * lead2.loc[0].get(i) / 200))),
                       (int(startX + i / 2.5),
                        int(textSpace + baseY - (gridSize * 2 * lead2.loc[0].get(i+1) / 200))),
                       (0, 0, 0), thickness)

    # Lead 3 - III
    baseY = margin + gridSize * 15
    startX = margin
    for i in range(1, partial_lead):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5),
                        int(textSpace + baseY - (gridSize * 2 * lead3.loc[0].get(i) / 200))),
                       (int(startX + i / 2.5),
                        int(textSpace + baseY - (gridSize * 2 * lead3.loc[0].get(i+1) / 200))),
                       (0, 0, 0), thickness)

    # Lead 4 - aVR
    baseY = margin + gridSize * 3
    startX = margin + (img_col - margin * 2 - 80) / 4
    for i in range(1, partial_lead):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead4.loc[0].get(partial_lead + i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead4.loc[0].get(partial_lead + i + 1) / 200))),
                       (0, 0, 0), thickness)

    # Lead 5 - aVL
    baseY = margin + gridSize * 9
    startX = margin + (img_col - margin * 2 - 80) / 4
    for i in range(1, partial_lead):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead5.loc[0].get(partial_lead + i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead4.loc[0].get(partial_lead + i + 1) / 200))),
                       (0, 0, 0), thickness)

    # Lead 6 - aVF
    baseY = margin + gridSize * 15
    startX = margin + (img_col - margin * 2 - 80) / 4
    for i in range(1, partial_lead):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead6.loc[0].get(partial_lead + i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead6.loc[0].get(partial_lead + i + 1) / 200))),
                       (0, 0, 0), thickness)

    # Lead 7 - V1
    baseY = margin + gridSize * 3
    startX = margin + (img_col - margin * 2 - 80) / 4 * 2
    for i in range(1, partial_lead):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead7.loc[0].get(partial_lead * 2 + i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead7.loc[0].get(partial_lead * 2 + i + 1) / 200))),
                       (0, 0, 0), thickness)

    # Lead 8 - V2
    baseY = margin + gridSize * 9
    startX = margin + (img_col - margin * 2 - 80) / 4 * 2
    for i in range(1, partial_lead):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead8.loc[0].get(partial_lead * 2 + i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead8.loc[0].get(partial_lead * 2 + i + 1) / 200))),
                       (0, 0, 0), thickness)

    # Lead 9 - V3
    baseY = margin + gridSize * 15
    startX = margin + (img_col - margin * 2 - 80) / 4 * 2
    for i in range(1, partial_lead):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead9.loc[0].get(partial_lead * 2 + i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead9.loc[0].get(partial_lead * 2 + i - 1) / 200))),
                       (0, 0, 0), thickness)

    # Lead 10 - V4
    baseY = margin + gridSize * 3
    startX = margin + (img_col - margin * 2 - 80) / 4 * 3
    for i in range(1, partial_lead + 200 - 1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead10.loc[0].get(partial_lead * 3 + i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead10.loc[0].get(partial_lead * 3 + i + 1) / 200))),
                       (0, 0, 0), thickness)

    # Lead 11 - V5
    baseY = margin + gridSize * 9
    startX = margin + (img_col - margin * 2 - 80) / 4 * 3
    for i in range(1, partial_lead + 200 - 1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead11.loc[0].get(partial_lead * 3 + i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead11.loc[0].get(partial_lead * 3 + i + 1) / 200))),
                       (0, 0, 0), thickness)

    # Lead 12 - V6
    baseY = margin + gridSize * 15
    startX = margin + (img_col - margin * 2 - 80) / 4 * 3
    for i in range(1, partial_lead + 200 - 1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead12.loc[0].get(partial_lead * 3 + i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead12.loc[0].get(partial_lead * 3 + i +1 ) / 200))),
                       (0, 0, 0), thickness)

    # Long 10s
    baseY = margin + gridSize * 21
    startX = margin
    for i in range(1, int(5200 - 1)):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5),
                        int(textSpace + baseY - (gridSize * 2 * lead2.loc[0].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead2.loc[0].get(i + 1) / 200))),
                       (0, 0, 0), thickness)
    # Export IMG
    file_name = df_target_info['filename'][0]
    # file_name = file_name.replace(':','_')
    cv2.imwrite(save_path + file_name + '.jpg', img)
