from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import os
import numpy as np


video = cv2.VideoCapture(0)
faces_detect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
with open('data/name.pkl',"rb") as f:
    LABLES= pickle.load(f)
with open('data/faces_data.pkl',"rb") as f:
    FACES= pickle.load(f)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABLES)
if not video.isOpened():
    print("Camera not opened")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faces_detect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        crop = frame[y:y+h, x:x+w] 
        resize = cv2.resize(crop, (50,50)).flatten().reshape(1,-1)
        output = knn.predict(resize)
        cv2.putText(frame, str(output[0]), (x,y-15),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,255), 2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
