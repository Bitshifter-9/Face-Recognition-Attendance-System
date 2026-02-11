import cv2
import pickle
import os
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
    if k==ord('q') or len(faces_data)==100:
        break
video.release()
cv2.destroyAllWindows()

faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(100,-1)

if "name.pkl" not in os.listdir('data/'):
    names=[name]*100
    with open('data/name.pkl',"wb") as f:
        pickle.dump(names,f)
else:
    with open('data/name.pkl',"rb") as f:
        names= pickle.load(f)
    names=names+[name]*100
    with open('data/name.pkl',"wb") as f:
        pickle.dump(names,f)

if "faces_data.pkl" not in os.listdir('data/'):
    
    with open('data/faces_data.pkl',"wb") as f:
        pickle.dump(faces_data,f)
else:
    with open('data/faces_data.pkl',"rb") as f:
        faces= pickle.load(f)
    faces=np.append(faces,faces_data,axis=0)
    with open('data/faces_data.pkl',"wb") as f:
        pickle.dump(faces,f)

