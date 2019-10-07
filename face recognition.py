#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 00:52:29 2019

@author: hargunasood
"""

import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

id = 0
names = ["Unknown", "Harguna Sood", "Novak Djokovic", "Eshaan"]

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/Users/hargunasood/trainer/trainer.yml')

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cap.read()
    #img2 = img
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(100, 100)
        #maxSize=(100,100)
    )
    eyes = eyeCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,#1.08,
        minNeighbors=13,     
        minSize=(20, 20),
        maxSize=(50,50)
            )
    
    smiles = smileCascade.detectMultiScale(
        gray,     
        scaleFactor=1.8,
        minNeighbors=26,  
        #minSize=(210,50),
        #maxSize=(270, 50)
            )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #blue
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        confidence = round(100-confidence, 0)
        if (confidence > 40):
            id = names[id]
        else:
            id = names[0]
            
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,0,0), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,0,0), 1)  
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w] 
        
    for (x,y,w,h) in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1) #green
        cv2.putText(img, "Eye", (x+5,y-5), font, 1, (0,255,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w] 
        
    """for (x,y,w,h) in smiles:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1) #red
        cv2.putText(img, "Lips", (x+5,y-5), font, 1, (0,0,255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]""" 
        
        
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

