'''
created by kwanglo @ 2021-05-18
requested by jspark
'''
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
save_path = 'D:\\DATA_STORAGE\\EXTRACTED\\IMG\\KUMC\\TEST\\12stack_noAmp\\'
pkl_files = os.listdir(pkl_path)


# Read info & ECG files
info_files = os.listdir(info_path)
# INFO
p = 7
infile = open(info_path + info_files[p], 'rb')
df_info = pickle.load(infile)
infile.close()
print(info_files[p])
# ECG
infile = open(pkl_path + pkl_files[0], 'rb')
df_total = pickle.load(infile)
infile.close()
print(pkl_files[0])

# Reference voltage 추가
arr_zero = np.zeros(50)
arr_one = np.ones(100) * 200
#arr_refer = pd.DataFrame(list(np.hstack((arr_zero, arr_one, arr_zero))))
arr_refer = list(np.hstack((arr_zero, arr_one, arr_zero)))

df_reference = list()
for r in tqdm(range(0, len(df_total))):
    df_reference.append(arr_refer)
df_reference = pd.DataFrame(df_reference)

col_list = []
for c in range(0, len(arr_refer)):
    col_list.append(5000 + c)

df_reference.columns = col_list
#############################################################################
# Read ECG
df_total = pd.concat([df_total,df_reference],axis=1)
for i in range(1,len(pkl_files)):
    infile = open(pkl_path + pkl_files[i], 'rb')
    df_temp = pickle.load(infile)
    infile.close()

    df_temp = pd.concat([df_temp, df_reference], axis=1)
    df_total = df_total.append(df_temp)
    print(pkl_files[i])
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
    img = np.zeros((3080,2180,3), np.uint8)
    img.fill(255)
    # 여기서 색상인 (255,0,0) 에 대해서는 RGB가 아닌 BGR 순서임.

    # Set features - 기본 크기나 여백은 그대로 (수정완료-2021/05/14 kwanglo)
    margin = 50
    gridSize = 40
    offset = 300
    x_offset = 50
    y_offset = 100
    textSpace = 200
    # 전체 크기 조정
    img_col = 2180
    img_row = 3080

    # Set grid
    # 그리드 갯수 수정 (수정완료-2021/05/14 kwanglo)
    for y_pos in range(0,72):
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
    fontScale = 0.5
    fontColor = (0,0,0)
    lineType = 2
    # Set writing position
    x_lead_std = int(margin)
    y_lead_std = int(textSpace + margin + 30)

    # Write leadNum
    img = cv2.putText(img,'I',(int(x_lead_std/2), y_lead_std),font,fontScale,fontColor,lineType)
    img = cv2.putText(img,'II',(int(x_lead_std/2), y_lead_std + gridSize * 6),font,fontScale,fontColor,lineType)
    img = cv2.putText(img,'III',(int(x_lead_std/2), y_lead_std + gridSize * 12),font,fontScale,fontColor,lineType)
    img = cv2.putText(img,'aVR',(int(x_lead_std/2), y_lead_std + gridSize * 18),font,fontScale,fontColor,lineType)
    img = cv2.putText(img,'aVL',(int(x_lead_std/2), y_lead_std + gridSize * 24),font,fontScale,fontColor,lineType)
    img = cv2.putText(img,'aVF',(int(x_lead_std/2), y_lead_std + gridSize * 30),font,fontScale,fontColor,lineType)
    img = cv2.putText(img,'V1',(int(x_lead_std/2), y_lead_std + gridSize * 36),font,fontScale,fontColor,lineType)
    img = cv2.putText(img,'V2',(int(x_lead_std/2), y_lead_std + gridSize * 42),font,fontScale,fontColor,lineType)
    img = cv2.putText(img,'V3',(int(x_lead_std/2), y_lead_std + gridSize * 48),font,fontScale,fontColor,lineType)
    img = cv2.putText(img,'V4',(int(x_lead_std/2), y_lead_std + gridSize * 54),font,fontScale,fontColor,lineType)
    img = cv2.putText(img,'V5',(int(x_lead_std/2), y_lead_std + gridSize * 60),font,fontScale,fontColor,lineType)
    img = cv2.putText(img,'V6',(int(x_lead_std/2), y_lead_std + gridSize * 66),font,fontScale,fontColor,lineType)

    # Write info
    info_names = ['filename', 'patient_id', 'gender', 'age',
                  'hr', 'date', 'Statement', 'temp_Label',
                  'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected', 'PAxis',
                  'RAxis', 'TAxis'
                  ]
    info_list = list(df_target_info[info_names].loc[0])

    # Preprocess
    # gender
    if info_list[2] == '1':
        gender = 'Male'
    elif info_list[2] == '2':
        gender = 'Female'
    elif  info_list[2] == '3':
        gender = 'Unknown'
    else:
        gender = 'Invalid'

    # Write first block
    img = cv2.putText(img, 'Date: ' + str(info_list[5]),(margin + 10, 35),font, 0.6,fontColor, lineType)
    img = cv2.putText(img, 'Patient ID: ' + str(info_list[1]), (margin + 10, 35 + 1 * 20), font, 0.6, fontColor, lineType)
    img = cv2.putText(img, 'Gender: ' + gender, (margin + 10, 35 + 2 * 20), font, 0.6, fontColor, lineType)
    img = cv2.putText(img, 'Age: ' + str(info_list[3]), (margin + 10, 35 + 3 * 20), font, 0.6, fontColor, lineType)

    # Write second block
    img = cv2.putText(img, 'Heart rate: ' + str(info_list[4]),(margin + 300, 35),font, 0.6,fontColor, lineType)
    img = cv2.putText(img, 'PR Interval: ' + str(info_list[8]), (margin + 300, 35 + 1 * 20), font, 0.6, fontColor, lineType)
    img = cv2.putText(img, 'QRS Duration: ' + str(info_list[9]), (margin + 300, 35 + 2 * 20), font, 0.6, fontColor, lineType)
    img = cv2.putText(img, 'QT Interval: ' + str(info_list[10]), (margin + 300, 35 + 3 * 20), font, 0.6, fontColor, lineType)

    # Write Third Block
    img = cv2.putText(img, 'QT corrected: ' + str(info_list[11]),(margin + 600, 35),font, 0.6,fontColor, lineType)
    img = cv2.putText(img, 'P axis: ' + str(info_list[12]), (margin + 600, 35 + 1 * 20), font, 0.6, fontColor, lineType)
    img = cv2.putText(img, 'R axis: ' + str(info_list[13]), (margin + 600, 35 + 2 * 20), font, 0.6, fontColor, lineType)
    img = cv2.putText(img, 'T axis: ' + str(info_list[14]), (margin + 600, 35 + 3 * 20), font, 0.6, fontColor, lineType)

    # Write statement
    statement_list = df_target_info['Statement'][0].split(', ')
    for state in range(0, len(statement_list)):
        img = cv2.putText(img, statement_list[state],
                          (int(margin + 1200), 35 + state * 20), font, 0.6, fontColor,
                          lineType)

    # Data
    thickness = 2
    #Lead1 I
    baseY = margin + gridSize * 3
    startX = margin
    for i in range(0, len(lead2.loc[imgNum])-1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead1.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead1.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead2 II
    baseY = margin + gridSize * 9
    startX = margin
    for i in range(0, len(lead2.loc[imgNum])-1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead2.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead2.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead3 III
    baseY = margin + gridSize * 15
    startX = margin
    for i in range(0, len(lead3.loc[imgNum])-1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead3.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead3.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead4 aVR
    baseY = margin + gridSize * 21
    startX = margin
    for i in range(0, len(lead3.loc[imgNum])-1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead4.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead4.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead5 aVL
    baseY = margin + gridSize * 27
    startX = margin
    for i in range(0, len(lead3.loc[imgNum])-1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead5.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead5.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead6 aVF
    baseY = margin + gridSize * 33
    startX = margin
    for i in range(0, len(lead3.loc[imgNum])-1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead6.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead6.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead7 V1
    baseY = margin + gridSize * 39
    startX = margin
    for i in range(0, len(lead3.loc[imgNum])-1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead7.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead7.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)
    #Lead8 V2
    baseY = margin + gridSize * 45
    startX = margin
    for i in range(0, len(lead3.loc[imgNum])-1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead8.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead8.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead9 V3
    baseY = margin + gridSize * 51
    startX = margin
    for i in range(0, len(lead3.loc[imgNum])-1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead9.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead9.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead10 V4
    baseY = margin + gridSize * 57
    startX = margin
    for i in range(0, len(lead3.loc[imgNum])-1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead10.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead10.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead11 V5
    baseY = margin + gridSize * 63
    startX = margin
    for i in range(0, len(lead3.loc[imgNum])-1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead11.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead11.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)

    #Lead12 V6
    baseY = margin + gridSize * 69
    startX = margin
    for i in range(0, len(lead3.loc[imgNum])-1):
        img = cv2.line(img,
                       (int(startX + (i - 1) / 2.5), int(textSpace + baseY - (gridSize * 2 * lead12.loc[imgNum].get(i) / 200))),
                       (int(startX + i / 2.5), int(textSpace + baseY - (gridSize * 2 * lead12.loc[imgNum].get(i + 1) / 200))),
                       (0,0,0), thickness)
    # 그려진 파일을 보여준다
    file_name = df_lead1['filename'][imgNum]
    #file_name = file_name.replace(':', '_')
    cv2.imwrite(save_path+file_name+'.jpg', img)
    #print('IMG done '+df_lead1['filename'][imgNum]+' / ' + 'RUN Time: ', stop - start)
