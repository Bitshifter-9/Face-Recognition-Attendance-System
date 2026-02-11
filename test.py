from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import os
import numpy as np
import csv
import time

from datetime import datetime


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
COL_NAME=['Name',"Time"]

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

        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        is_file=os.path.isfile("attendance/Attendance_"+date+".csv")
        cv2.putText(frame, str(output[0]), (x,y-15),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,255), 2)
        attendance=[str(output[0]),timestamp]
    cv2.imshow("frame", frame)
    k=cv2.waitKey(1)
    if k==ord('o') or k==ord("O"):
        # Database connection
        from sqlalchemy import create_engine
        import pandas as pd
        
        # Connect to database (replace with your actual database URL)
        # You should set this in your environment variables for security
        # For local testing, you can use a local postgres url like: postgresql://user:password@localhost/dbname
        DATABASE_URL = os.environ.get('DATABASE_URL')
        
        if DATABASE_URL:
            try:
                engine = create_engine(DATABASE_URL)
                attendance_db = [str(output[0]), timestamp, date]
                columns_db = ['Name', 'Time', 'Date']
                
                df = pd.DataFrame([attendance_db], columns=columns_db)
                df.to_sql('attendance', engine, if_exists='append', index=False)
                print(f"Attendance marked for {output[0]} in Database")
            except Exception as e:
                print(f"Error saving to database: {e}")
        else:
             print("DATABASE_URL not set! Falling back to CSV.")
             if is_file:
                 with open("attendance/Attendance_"+date+".csv","+a") as f:
                     writer=csv.writer(f)
                     writer.writerow(attendance)
                 f.close()
             else:
                 with open("attendance/Attendance_"+date+".csv","+a") as f:
                     writer=csv.writer(f)
                     writer.writerow(COL_NAME)
                     writer.writerow(attendance)
                 f.close()
            
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
