#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:40:30 2019

@author: hargunasood
"""

import cv2
import numpy as np
from PIL import Image
import os

path = "/Users/hargunasood/Harguna_Face_Dataset/"

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[0])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.save('/Users/hargunasood/trainer/trainer.yml') 
# find . -name '.DS_Store' -type f -delete
