import numpy as np
import argparse
import os
import imutils
import cv2
import random
import csv
#type 1 - Meta     type 2 - Test    Type3 - Train
def augment(type,image,file,_list):
    angle=0
    for i in range (0,2):#sput to 4
        rotated_image = imutils.rotate(image, angle)
        angle=(angle+90)%360
        #crop - scale in % way from borders
        for j in range (0,2):#normally 4
            h, w, _ = image.shape
            ha=int((random.randint(0,5)/100)*h)
            wa=int((random.randint(0,5)/100)*w)
            hz=int((random.randint(95,100)/100)*h)
            wz=int((random.randint(95,100)/100)*w)
            scaled_rotated_image=rotated_image[ha:hz,wa:wz]
            if(type==1):     
                cv2.imwrite( 'augmented/meta/'+file.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png', scaled_rotated_image)
                tmp_list=list();
                tmp_list.append(['meta/'+file.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',str(d[1]),str(d[2]),str(d[3]),str(d[4])])
                _list.extend(tmp_list);
            elif(type==2):
                cv2.imwrite( 'augmented/test/'+file.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png', scaled_rotated_image)            
                tmp_list=list();
                tmp_list.append(['test/'+file.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',str(d[6])])
                _list.extend(tmp_list);
            elif(type==3):
                cv2.imwrite( 'augmented/train/'+d[7].split('/')[1]+'/'+file.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png', scaled_rotated_image)
                tmp_list=list();
                tmp_list.append(['train/'+d[7].split('/')[1]+'/'+file.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',str(d[6])])
                _list.extend(tmp_list);       
def load_csv(path):
    with open(path, newline='') as csvfile:
        return list(csv.reader(csvfile))    
def save_csv(path):
    with open(path, 'w') as newfile:
        for i in _list:
            wr = csv.writer(newfile, quoting=csv.QUOTE_ALL)
            wr.writerow(i)    
#check for required destination folders
if not os.path.exists("augmented"):
    os.mkdir("augmented")
if not os.path.exists("augmented/train"):
    os.mkdir("augmented/train/")
if not os.path.exists("augmented/meta"):
    os.mkdir("augmented/meta/")
if not os.path.exists("augmented/test"):
    os.mkdir("augmented/test/")
#meta folder
data=load_csv('Meta.csv')
# #augmentation meta folder
_list=list()
for d in data[1:]:
    file=d[0].split('/')[1]
    image = cv2.imread(d[0])
    augment(1,image,file,_list)
save_csv("augmented/Meta_augmented.csv")
print("done meta")
_list=list()
#test folders
data=load_csv('Test.csv')
for d in data[1:]:
    file=d[7].split('/')[1]
    image = cv2.imread(d[7])
    augment(2,image,file,_list)
save_csv("augmented/Test_augmented.csv")
print("done test")
_list=list()
# #train folders
stop=1
data=load_csv('Train.csv')
for dirName, subdirList, fileList in os.walk("train/"):
    if stop==1: 
        stop=2
        for subd in subdirList:
            if not os.path.exists("augmented/train/"+str(subd)):
                print("class",subd)
                os.mkdir("augmented/train/"+str(subd))
            for d in data[1:]:
                file=d[7].split('/')[2]
                image = cv2.imread(d[7])
                augment(3,image,file,_list)
save_csv("augmented/Train_augmented.csv")