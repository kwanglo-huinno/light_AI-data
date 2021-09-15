import pickle
import pandas as pd
import numpy as np
import os
import timeit
import cv2
import sklearn
from sklearn import preprocessing
from tqdm import tqdm

info_path = 'D:\\DATA_STORAGE\\EXTRACTED\\PKL\\KUMC\\'
pkl_path = 'D:\\DATA_STORAGE\\EXTRACTED\\PKL\\KUMC\\ECG_12lead_noAmp_filtered\\'
save_path = 'D:\\DATA_STORAGE\\EXTRACTED\\IMG\\KUMC\\TEST\\4stack_noAmp_test\\'
pkl_files = os.listdir(pkl_path)

# Read info & ECG files
info_files = os.listdir(info_path)
# INFO
p = 6
infile = open(info_path + info_files[p], 'rb')
df_info = pickle.load(infile)
infile.close()
print(info_files[p])
# ECG
infile = open(pkl_path + pkl_files[0], 'rb')
df_total = pickle.load(infile)
infile.close()
print(pkl_files[0])
for i in range(1,len(pkl_files)):
    infile = open(pkl_path + pkl_files[i], 'rb')
    df_temp = pickle.load(infile)
    infile.close()
    df_total = df_total.append(df_temp)
    print(pkl_files[i])

# Reference voltage 추가
arr_zero = np.zeros(50)
arr_one = np.ones(100)
arr_refer = pd.DataFrame(list(np.hstack((arr_zero,arr_one,arr_zero))))

# I
df_lead1 = df_total[df_total['lead']=='lead1']
# II
df_lead2 = df_total[df_total['lead']=='lead2']
# III
df_lead3 = df_total[df_total['lead']=='lead3']
# aVR
df_lead4 = df_total[df_total['lead']=='lead4']
# aVL
df_lead5 = df_total[df_total['lead']=='lead5']
# aVF
df_lead6 = df_total[df_total['lead']=='lead6']
# V1
df_lead7 = df_total[df_total['lead']=='lead7']
# V2
df_lead8 = df_total[df_total['lead']=='lead8']
# V3
df_lead9 = df_total[df_total['lead']=='lead9']
# V4
df_lead10 = df_total[df_total['lead']=='lead10']
# V5
df_lead11 = df_total[df_total['lead']=='lead11']
# V6
df_lead12 = df_total[df_total['lead']=='lead12']

# Get df_ecg
info_idx = 2
lead1 = df_lead1.iloc[:,info_idx:]
lead1 = df_lead1.iloc[:,info_idx:]
lead2 = df_lead2.iloc[:,info_idx:]
lead3 = df_lead3.iloc[:,info_idx:]
lead4 = df_lead4.iloc[:,info_idx:]
lead5 = df_lead5.iloc[:,info_idx:]
lead6 = df_lead6.iloc[:,info_idx:]
lead7 = df_lead7.iloc[:,info_idx:]
lead8 = df_lead8.iloc[:,info_idx:]
lead9 = df_lead9.iloc[:,info_idx:]
lead10 = df_lead10.iloc[:,info_idx:]
lead11 = df_lead11.iloc[:,info_idx:]
lead12 = df_lead12.iloc[:,info_idx:]

###### DRAW IMG ######
# set
imgNum = 0
for imgNum in tqdm(range(0,10)): #len(lead1)
    df_target_info = df_info[df_info['filename']==df_lead1['filename'][imgNum]].reset_index(drop=True)
    # np.zeros(y,x,3 color channel)
    # 크기 수정 (수정완료-2021/05/14 kwanglo)
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
    # 그리드 갯수 수정 (수정완료-2021/05/14 kwanglo)
    for y_pos in range(0,25):
        # start(x,y) / end(x,y) / color / px
        img = cv2.line(img,
                       (margin,textSpace+margin+gridSize*y_pos), # Start
                       (img_col-margin,textSpace+margin+gridSize*y_pos), # End
                       (100,100,255),1) # Color, px
    for x_pos in range(0,53):
        img = cv2.line(img,
                       (margin+gridSize*x_pos,textSpace+margin), # Start
                       (margin+gridSize*x_pos,img_row-margin+25), # End
                       (100,100,255),1) # Color, px

    # Set font features
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale = 0.8
    fontColor = (0,0,0)
    lineType = 2
    # Set writing position
    x_lead_std = int(margin + 10)
    y_lead_std = int(textSpace + margin + 30)
    x_lead_augmented = int(margin + 10 + (img_col - margin * 2 - 80) / 4)

    # Write leadNum
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
        img = cv2.putText(img, info_names[info] + ': ' + str(info_list[info]), (margin + 10, 35 + info * 20), font, 0.6,
                          fontColor, lineType)
    # Write statement
    statement_list = df_target_info['Statement'][0].split(',')
    for state in range(0, len(statement_list)):
        img = cv2.putText(img, statement_list[state],
                          (int(margin + (img_col - margin * 2 - 80) / 4 * 2), 35 + state * 20), font, 0.6, fontColor,
                          lineType)

    # Data
    thickness = 2
    partial_lead = int(5000 / 4)

    #Lead1 I
    baseY = margin + gridSize * 3
    startX = margin
    for i in range(0, partial_lead):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead1.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead1.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead2 II
    baseY = margin + gridSize * 9
    startX = margin
    for i in range(0, partial_lead):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead2.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead2.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead3 III
    baseY = margin + gridSize * 15
    startX = margin
    for i in range(0, partial_lead):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead3.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead3.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead4 aVR
    baseY = margin + gridSize * 3
    startX = margin + (img_col - margin * 2 - 80) / 4
    for i in range(partial_lead, partial_lead * 2):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead4.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead4.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead5 aVL
    baseY = margin + gridSize * 9
    startX = margin + (img_col - margin * 2 - 80) / 4
    for i in range(partial_lead, partial_lead * 2):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead5.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead5.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead6 aVF
    baseY = margin + gridSize * 15
    startX = margin + (img_col - margin * 2 - 80) / 4
    for i in range(partial_lead, partial_lead * 2):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead6.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead6.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead7 V1
    baseY = margin + gridSize * 3
    startX = margin + (img_col - margin * 2 - 80) / 4 * 2
    for i in range(partial_lead * 2, partial_lead * 3):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead7.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead7.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)
    #Lead8 V2
    baseY = margin + gridSize * 9
    startX = margin + (img_col - margin * 2 - 80) / 4 * 2
    for i in range(partial_lead * 2, partial_lead * 3):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead8.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead8.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead9 V3
    baseY = margin + gridSize * 15
    startX = margin + (img_col - margin * 2 - 80) / 4 * 2
    for i in range(partial_lead * 2, partial_lead * 3):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead9.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead9.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead10 V4
    baseY = margin + gridSize * 3
    startX = margin + (img_col - margin * 2 - 80) / 4 * 3
    for i in range(partial_lead * 3, partial_lead * 4 - 1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead10.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead10.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead11 V5
    baseY = margin + gridSize * 9
    startX = margin + (img_col - margin * 2 - 80) / 4 * 3
    for i in range(partial_lead * 3, partial_lead * 4 - 1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead11.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead11.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead12 V6
    baseY = margin + gridSize * 15
    startX = margin + (img_col - margin * 2 - 80) / 4 * 3
    for i in range(partial_lead * 3, partial_lead * 4 - 1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead12.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead12.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    # Lead2 - long
    baseY = margin + gridSize * 21
    startX = margin
    for i in range(0, int(5000-1)):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead2.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead2.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)
    # 그려진 파일을 보여준다
    file_name = df_lead1['filename'][imgNum]
    #file_name = file_name.replace(':', '_')
    cv2.imwrite(save_path+file_name+'.jpg', img)
    #print('IMG done '+df_lead1['filename'][imgNum]+' / ' + 'RUN Time: ', stop - start)
