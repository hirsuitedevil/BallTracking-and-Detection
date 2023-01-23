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


ball_df = pd.DataFrame(columns=['frame','x','y','w','h'])
frames = os.listdir('frames/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

for idx in range(len(frames)):
    
    img= cv2.imread('frames/' + frames[idx])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(25, 25),0)
    _ , mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    dir = 'C:/Users/Bhavya/Desktop/Sel_topic_Hackathon/patch'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    num=20
    cnt=0
    df = pd.DataFrame(columns=['frame','x','y','w','h'])
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])

        numer=min([w,h])
        denom=max([w,h])
        ratio=numer/denom

        if(x>=num and y>=num):
            xmin, ymin= x-num, y-num
            xmax, ymax= x+w+num, y+h+num
        else:
            xmin, ymin= x,y
            xmax, ymax= x+w, y+h

        if(ratio>=0.5):    
            #print(cnt,x,y,w,h,ratio)
            df.loc[cnt,'frame'] = frames[idx]
            df.loc[cnt,'x']=x
            df.loc[cnt,'y']=y
            df.loc[cnt,'w']=w
            df.loc[cnt,'h']=h
            
            cv2.imwrite("patch/"+str(cnt)+".png",img[ymin:ymax,xmin:xmax])
            cnt=cnt+1
    
    
    files=os.listdir('patch/')    
    if(len(files)>0):
    
        files.sort(key=lambda f: int(re.sub('\D', '', f)))

        test=[]
        for file in files:
            img=cv2.imread('patch/'+file,0)
            img=cv2.resize(img,(25,25))
            test.append(img)

        test = np.array(test)

        test = test.reshape(len(test),-1)
        y_pred = rfc.predict(test)
        prob=rfc.predict_proba(test)

        if 0 in y_pred:
            ind = np.where(y_pred==0)[0]
            proba = prob[:,0]
            confidence = proba[ind]
            confidence = [i for i in confidence if i>0.7]
            if(len(confidence)>0):

                maximum = max(confidence)
                ball_file=files[list(proba).index(maximum)]

                img= cv2.imread('patch/'+ball_file)
                cv2.imwrite('ball/'+str(frames[idx]),img)

                no = int(ball_file.split(".")[0])
                ball_df.loc[idx]= df.loc[no]
            else:
                ball_df.loc[idx,'frame']=frames[idx]

        else:
            ball_df.loc[idx,'frame']=frames[idx]


ball_df.dropna(inplace=True)
print(ball_df)

files = ball_df['frame'].values
num=10
for idx in range(len(files)):
    
    #draw contours 
    img = cv2.imread('frames/'+files[idx])
    x=ball_df.loc[idx,'x']
    y=ball_df.loc[idx,'y']
    w=ball_df.loc[idx,'w']
    h=ball_df.loc[idx,'h']
    
    xmin=x-num
    ymin=y-num
    xmax=x+w+num
    ymax=y+h+num

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    cv2.imwrite("frames/"+files[idx],img)


frames = os.listdir('frames/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

frame_array=[]

for i in range(len(frames)):
    #reading each files
    img = cv2.imread('frames/'+frames[i])
    height, width, layers = img.shape
    size = (width,height)
    #inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter('20.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
 
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()