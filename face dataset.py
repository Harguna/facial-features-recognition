#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:31:52 2019

@author: hargunasood
"""

import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
user_id = input("User ID: ")
count = 0
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=2,     
        minSize=(100, 100)
    )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        cv2.imwrite("/Users/hargunasood/Harguna_Face_Dataset/" + str(user_id) + "." + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 200: # Take 30 face sample and stop video
         break

cam.release()
cv2.destroyAllWindows()
