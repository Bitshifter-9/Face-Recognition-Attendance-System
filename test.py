from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import os
import numpy as np
import csv
import time

from datetime import datetime

mode=input("Choose mode (knn/arcface): ").strip().lower()

video = cv2.VideoCapture(0)
faces_detect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

if mode=="knn":
    with open('data/name.pkl',"rb") as f:
        LABLES= pickle.load(f)
    with open('data/faces_data.pkl',"rb") as f:
        FACES= pickle.load(f)
    knn=KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES,LABLES)
    print("KNN model loaded.")

elif mode=="arcface":
    from deepface import DeepFace
    ARCFACE_DB="faces_db"
    if not os.path.exists(ARCFACE_DB) or len(os.listdir(ARCFACE_DB))==0:
        print("No faces in faces_db/. Run add_faces.py with arcface mode first.")
        exit()
    print("ArcFace model ready.")

else:
    print("Invalid mode. Use 'knn' or 'arcface'.")
    exit()

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

    if mode=="knn":
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

    elif mode=="arcface" and len(faces)>0:
        try:
            results=DeepFace.find(
                img_path=frame,
                db_path=ARCFACE_DB,
                model_name="ArcFace",
                detector_backend="opencv",
                distance_metric="cosine",
                enforce_detection=False,
                silent=True,
            )
            name="Unknown"
            confidence=0
            if len(results)>0 and len(results[0])>0:
                match_df=results[0]
                match_df=match_df[match_df['distance']<0.40]
                if len(match_df)>0:
                    best=match_df.iloc[0]
                    name=os.path.splitext(os.path.basename(best['identity']))[0].replace("_"," ")
                    confidence=round((1-best['distance'])*100,1)

            for (x,y,w,h) in faces:
                if name!="Unknown":
                    color=(0,200,0)
                    label=f"{name} ({confidence}%)"
                else:
                    color=(0,0,255)
                    label="Unknown"
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,label,(x,y-15),cv2.FONT_HERSHEY_COMPLEX,0.8,color,2)

            ts=time.time()
            date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
            is_file=os.path.isfile("attendance/Attendance_"+date+".csv")
            attendance=[name,timestamp]
        except Exception as e:
            print(f"ArcFace error: {e}")

    cv2.imshow("frame", frame)
    k=cv2.waitKey(1)
    if k==ord('o') or k==ord("O"):
        from sqlalchemy import create_engine
        import pandas as pd
        
        DATABASE_URL = os.environ.get('DATABASE_URL')
        
        if DATABASE_URL:
            try:
                engine = create_engine(DATABASE_URL)
                if mode=="knn":
                    att_name=str(output[0])
                else:
                    att_name=attendance[0]
                attendance_db = [att_name, timestamp, date]
                columns_db = ['Name', 'Time', 'Date']
                
                df = pd.DataFrame([attendance_db], columns=columns_db)
                df.to_sql('attendance', engine, if_exists='append', index=False)
                print(f"Attendance marked for {att_name} in Database")
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
