
import numpy as np
import cv2  as cv
import  matplotlib.pyplot as plt
import os

faceCascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
# Read image from your local file system

for i in range(1,51):
    
    # ele vai criar uma pasta e vai entrar em cada imagem e cortar
    if len(str(i)) == 1:
        numero_pasta="00" + str(i)
    if len(str(i)) == 2:
        numero_pasta="0" + str(i)
    
   
    os.mkdir('./'+numero_pasta)
    for v in range (1,6):
        #formatar o numero
       
        pasta='C:/Users/camil/Desktop/Face Database part1/'+numero_pasta+'/'+numero_pasta+'_expresion_frown_'+str(v)+'.bmp'
        original_image = cv.imread(pasta,0)
        original_image_notgray = cv.imread(pasta,1)

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
            sub_face = original_image_notgray[y:y+h, x:x+w]
            cv.imwrite('./'+numero_pasta+'/'+numero_pasta+'_crop_'+str(v)+'.bmp',sub_face)

