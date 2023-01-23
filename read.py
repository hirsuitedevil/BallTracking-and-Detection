import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import re

#listing down all the file names
frames = os.listdir('frames/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))
#print(frames)

#reading frames
images=[]
for i in frames:
    img = cv2.imread('frames/'+i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(25,25),0)
    images.append(img)

images=np.array(images)

nonzero=[]
for i in range((len(images)-1)):
    
    mask = cv2.absdiff(images[i],images[i+1])
    _ , mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    num = np.count_nonzero((mask.ravel()))
    nonzero.append(num)
    
    
x = np.arange(0,len(images)-1)
y = nonzero

plt.figure(figsize=(20,4))
plt.scatter(x,y)

threshold = 10 * 10e2
for i in range(len(images)-1):
    if(nonzero[i]>threshold): 
        scene_change_idx = i
        break
        
frames = frames[:(scene_change_idx+1)]
cnt=0

dir = 'C:/Users/Bhavya/Desktop/Sel_topic_Hackathon/patch'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))
    
for ind in range(len(frames)):
    img= cv2.imread('frames/' + frames[ind])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(25,25),0)

    plt.figure(figsize=(5,10))
    plt.imshow(gray,cmap='gray')
    _ , mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    plt.figure(figsize=(5,5))
    plt.imshow(mask,cmap='gray')

    contours, image = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = np.copy(gray)
    cv2.drawContours(img_copy, contours, -1, (0,255,0), 3)
    plt.imshow(img_copy, cmap='gray')

    num=20
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        
        numer=min([w,h])
        denom=max([w,h])
        ratio=numer/denom

        if(x>=num and y>=num):
            xmin, ymin= x-num, y-num
            xmax, ymax= x+w+num, y+h+num
        else:
            xmin, ymin=x, y
            xmax, ymax=x+w, y+h

        if(ratio>=0.5 and ((w<=10) and (h<=10)) ):    
            print(cnt,x,y,w,h,ratio)
            cv2.imwrite("patch/"+str(cnt)+".png",img[ymin:ymax,xmin:xmax])
            cnt=cnt+1