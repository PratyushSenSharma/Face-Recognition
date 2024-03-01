import cv2
from cv2 import cvtColor
from random import randrange

#Load some pre-trained data on faxce frontals from opencv(haar cascade algorithm)
trained_face_data=cv2.CascadeClassifier('harrcascade_frontalface_default.xml')

#Choose an image to detect faces in
img=cv2.imread('RDJ.jpg')

#Must convert to grayscale
grayscaled_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect Faces
face_coordinates= trained_face_data.detectMultiScale(grayscaled_img)

#Draw rectangle around faces
for (x,y,w,h) in face_coordinates:
  cv2.rectangle(img, (x, y), (x+h, y+w), (randrange(256), randrange(256), randrange(256)), 2)



cv2.imshow('clever programmer face detector', img)
cv2.waitKey()

print('code completed')