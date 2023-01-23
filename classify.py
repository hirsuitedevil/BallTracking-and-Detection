import os
import cv2
import numpy as np
import pandas as pd
import re

folders=os.listdir('data/')

images=[]
labels= []
for folder in folders:
    files=os.listdir('data/'+folder)
    for file in files:
        img=cv2.imread('data/'+folder+'/'+file,0)
        img=cv2.resize(img,(25,25))
        images.append(img)
        x=0
        if(folder == 'ball'):
            x=1
        labels.append(x)

images = np.array(images)
features = images.reshape(len(images),-1)

from sklearn.model_selection import train_test_split
x_tr,x_val,y_tr,y_val = train_test_split(features,labels, test_size=0.2, stratify=labels,random_state=42)

from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier(max_depth=3) 
rfc.fit(x_tr,y_tr)

from sklearn.metrics import classification_report
y_pred = rfc.predict(x_val)
print(classification_report(y_val,y_pred))

