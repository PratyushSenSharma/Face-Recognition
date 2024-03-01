import cv2
from cv2 import cvtColor
from random import randrange

#Load some pre-trained data on faxce frontals from opencv(haar cascade algorithm)
trained_face_data=cv2.CascadeClassifier('D:\VS CODE\FaceRecognition\harrcascade_frontalface_default.xml')

# To capture video from webcam
webcam=cv2.VideoCapture(0)

# Iterate forever over frames
while True:
 #read current frame
  sucessfull_frame_read, frame = webcam.read()
   # Must convert to grayscale
  grayscaled_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  print('code completed') 

  # Detect Faces
  face_coordinates= trained_face_data.detectMultiScale(grayscaled_img)

  # Draw rectangle around faces
  for (x,y,w,h) in face_coordinates:
   cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 255, 0), 2)



  cv2.imshow('clever programmer face detector', frame)
  cv2.waitKey(1)

  print('code completed')