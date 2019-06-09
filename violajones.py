
import numpy as np
import cv2  as cv
import  matplotlib.pyplot as plt
import os


faceCascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
# Read image from your local file system
# fazer um foreach pelas imagens
original_image = cv.imread('eu_linkedin.jpg',0)

# Detect faces
faces = faceCascade.detectMultiScale(
original_image,
scaleFactor=1.1,
minNeighbors=5,
flags=cv.CASCADE_SCALE_IMAGE
)
# For each face
for (x, y, w, h) in faces: 
    # Draw rectangle around the face
    sub_face = original_image[y:y+h, x:x+w]
    cv.imwrite("crop1.jpg",sub_face)

