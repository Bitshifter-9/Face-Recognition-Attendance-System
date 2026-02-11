import streamlit as st
import pandas as pd 
import time
from datetime import datetime
import os 
from sqlalchemy import create_engine, text
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

def get_db_engine():
    if 'DATABASE_URL' in os.environ:
        return create_engine(os.environ['DATABASE_URL'])
    else:
        st.error("DATABASE_URL not set.")
        return None

engine = get_db_engine()

def load_faces_from_db():
    if not engine: return [], []
    try:
        query = text("SELECT name, face_encoding FROM registered_faces")
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        
        faces = []
        labels = []
        for row in result:
            name = row[0]
            face_encoding_bytes = row[1]
            face_encoding = pickle.loads(face_encoding_bytes)
            faces.append(face_encoding)
            labels.append(name)
        return np.array(faces), np.array(labels)
    except Exception as e:
        st.error(f"Error loading faces from DB: {e}")
        return [], []

def train_knn(faces, labels):
    if len(faces) == 0: return None
    n_neighbors = min(5, len(faces))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    if len(faces.shape) > 2:
        faces = faces.reshape(faces.shape[0], -1) 
    knn.fit(faces, labels)
    return knn

st.set_page_config(page_title="Attendance System", layout="wide")
st.title("Face Recognition Attendance System")

tab1, tab2, tab3 = st.tabs(["📷 Mark Attendance", "📝 Register New Face", "📊 View Records"])

with tab2:
    st.header("Register New Face")
    
    reg_name = st.text_input("Enter Name", placeholder="e.g., John Doe")
    reg_img_buffer = st.camera_input("Take a photo to register", key="register_cam")
    
    if st.button("Save Face"):
        if not reg_name:
            st.error("Please enter a name.")
        elif reg_img_buffer is None:
            st.error("Please take a photo.")
        else:
            bytes_data = reg_img_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            faces_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces = faces_detect.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                (x,y,w,h) = max(faces, key=lambda b: b[2] * b[3])
                crop = cv2_img[y:y+h, x:x+w]
                
                st.write("Generating 100 face samples for better accuracy...")
                progress_bar = st.progress(0)
                
                samples_generated = 0
                
                try:
                    with engine.connect() as conn:
                        for i in range(100):
                            if i < 5:
                                augmented_img = crop
                            else:
                                rows, cols, _ = crop.shape
                                
                                angle = np.random.uniform(-10, 10)
                                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                                augmented_img = cv2.warpAffine(augmented_img, M, (cols, rows))
                                
                                brightness = np.random.uniform(0.8, 1.2)
                                augmented_img = cv2.convertScaleAbs(augmented_img, alpha=brightness, beta=0)
                                
                                tx = np.random.uniform(-2, 2)
                                ty = np.random.uniform(-2, 2)
                                M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
                                augmented_img = cv2.warpAffine(augmented_img, M_shift, (cols, rows))

                            resize = cv2.resize(augmented_img, (50,50)).flatten().reshape(1,-1)
                            face_pickle = pickle.dumps(resize)
                            
                            query = text("INSERT INTO registered_faces (name, face_encoding) VALUES (:name, :encoding)")
                            conn.execute(query, {"name": reg_name, "encoding": face_pickle})
                            
                            samples_generated += 1
                            progress_bar.progress(samples_generated / 100)
                        
                        conn.commit()
                        
                    st.success(f"✅ Successfully registered **{reg_name}** with 100 samples!")
                    st.info("The model is now trained on these 100 samples.")
                    
                except Exception as e:
                    st.error(f"Error saving to DB: {e}")
            else:
                st.error("No face detected. Please try again.")
    
with tab1:
    st.header("Mark Attendance")
    
    if len(FACES) > 0:
        knn = train_knn(FACES, LABELS)
        
        img_file_buffer = st.camera_input("Take a photo to mark attendance", key="attendance_cam")
        
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            faces_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces = faces_detect.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                for (x,y,w,h) in faces:
                    crop = cv2_img[y:y+h, x:x+w]
                    resize = cv2.resize(crop, (50,50)).flatten().reshape(1,-1)
                    prediction = knn.predict(resize)
                    name = prediction[0]
                    
                    cv2.rectangle(cv2_img, (x,y), (x+w,y+h), (50,50,255), 2)
                    cv2.putText(cv2_img, name, (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
                    
                    ts = time.time()
                    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                    timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
                    
                    try:
                        attendance_data = pd.DataFrame([[name, timestamp, date]], columns=['Name', 'Time', 'Date'])
                        attendance_data.to_sql('attendance', engine, if_exists='append', index=False)
                        st.success(f"✅ Attendance marked for **{name}** at {timestamp}")
                    except Exception as e:
                        st.error(f"Error saving attendance: {e}")
                
                st.image(cv2_img, channels="BGR", caption="Processed Image")
            else:
                st.warning("No face detected. Please try again.")
    else:
        st.warning("No registered faces found. Please go to the 'Register New Face' tab.")