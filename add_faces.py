import cv2
import pickle
import numpy as np

video = cv2.VideoCapture(0)
faces_detect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
faces_data=[]
i=0
name=input("Enter your name: ")
while True:
    ref,frame=video.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= faces_detect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
       
        crop=frame[y:y+h,x:x+w,:]
        resize=cv2.resize(crop,(50,50))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resize)
        i+=1
        cv2.putText(frame,str(len(faces_data)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
    cv2.imshow("frame",frame)

    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
