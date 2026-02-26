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

@st.cache_resource
def load_deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except ImportError:
        return None

ARCFACE_DB_DIR = "faces_db"

def restore_arcface_faces_from_db():
    """Restore ArcFace face images from the database to the filesystem.
    This ensures faces persist across Streamlit Cloud restarts."""
    if not engine:
        return
    os.makedirs(ARCFACE_DB_DIR, exist_ok=True)
    try:
        # Ensure the arcface_faces table exists
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS arcface_faces (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    image_data BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.commit()
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name, image_data FROM arcface_faces")).fetchall()
        
        for row in result:
            name = row[0]
            image_data = row[1]
            if isinstance(image_data, memoryview):
                image_data = bytes(image_data)
            img_path = os.path.join(ARCFACE_DB_DIR, f"{name}.jpg")
            if not os.path.exists(img_path):
                with open(img_path, 'wb') as f:
                    f.write(image_data)
    except Exception as e:
        st.error(f"Error restoring ArcFace faces from DB: {e}")

def save_arcface_face_to_db(safe_name, img_path):
    """Save an ArcFace face image to the database for persistence."""
    if not engine:
        return
    try:
        with open(img_path, 'rb') as f:
            image_data = f.read()
        with engine.connect() as conn:
            # Remove old entry for this name if exists
            conn.execute(text("DELETE FROM arcface_faces WHERE name = :name"), {"name": safe_name})
            conn.execute(
                text("INSERT INTO arcface_faces (name, image_data) VALUES (:name, :image_data)"),
                {"name": safe_name, "image_data": image_data}
            )
            conn.commit()
    except Exception as e:
        st.error(f"Error saving ArcFace face to DB: {e}")

# Restore ArcFace faces from DB on startup
restore_arcface_faces_from_db()

def get_registered_arcface_names():
    os.makedirs(ARCFACE_DB_DIR, exist_ok=True)
    names = []
    for f in os.listdir(ARCFACE_DB_DIR):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            names.append(os.path.splitext(f)[0])
    return sorted(set(names))

st.set_page_config(page_title="Attendance System", layout="wide")
st.title("Face Recognition Attendance System")

st.sidebar.header("⚙️ Recognition Model")
mode = st.sidebar.radio("Select model:", ["KNN (Classic ML)", "ArcFace (Deep Learning)"])

if "KNN" in mode:
    st.sidebar.info("**KNN** — compares raw pixels, needs 100 photos per person, ~70-80% accuracy.")
    FACES, LABELS = load_faces_from_db()
else:
    st.sidebar.success("**ArcFace** — deep learning embeddings, needs 1 photo per person, 99.8% accuracy.")

tab1, tab2, tab3 = st.tabs(["📷 Mark Attendance", "📝 Register New Face", "📊 View Records"])

with tab2:
    st.header("Register New Face")
    
    if "KNN" in mode:
        st.caption("📌 KNN mode — 100 augmented samples will be generated")
        reg_name = st.text_input("Enter Name", placeholder="e.g., John Doe", key="knn_reg_name")
        reg_img_buffer = st.camera_input("Take a photo to register", key="register_cam_knn")
        
        if st.button("Save Face", key="knn_save"):
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
                        st.success(f"✅ Registered **{reg_name}** with 100 samples!")
                    except Exception as e:
                        st.error(f"Error saving to DB: {e}")
                else:
                    st.error("No face detected. Please try again.")
    else:
        st.caption("📌 ArcFace mode — only 1 photo needed")
        reg_name = st.text_input("Enter Name", placeholder="e.g., John Doe", key="arc_reg_name")
        reg_img_buffer = st.camera_input("Take a photo to register", key="register_cam_arc")
        
        if st.button("Save Face", key="arc_save"):
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
                    os.makedirs(ARCFACE_DB_DIR, exist_ok=True)
                    safe_name = reg_name.strip().replace(" ", "_")
                    img_path = os.path.join(ARCFACE_DB_DIR, f"{safe_name}.jpg")
                    cv2.imwrite(img_path, crop)
                    save_arcface_face_to_db(safe_name, img_path)
                    st.success(f"✅ Registered **{reg_name}** with ArcFace! (1 photo saved)")
                else:
                    st.error("No face detected. Please try again.")

        registered = get_registered_arcface_names()
        if registered:
            st.markdown("**Registered faces:** " + ", ".join(registered))

with tab3:
    st.header("Attendance Records")
    
    if engine:
        try:
            query = f"SELECT * FROM attendance" 
            df = pd.read_sql(query, engine)
            st.dataframe(df, width='stretch')
        except Exception as e:
            st.error(f"Error loading records: {e}")
    
with tab1:
    st.header("Mark Attendance")
    st.caption(f"🔍 Using **{mode}** model")
    
    if "KNN" in mode:
        if len(FACES) > 0:
            knn = train_knn(FACES, LABELS)
            
            img_file_buffer = st.camera_input("Take a photo to mark attendance", key="attendance_cam_knn")
            
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
                            st.success(f"✅ Attendance marked for **{name}** at {timestamp} (KNN)")
                        except Exception as e:
                            st.error(f"Error saving attendance: {e}")
                    
                    st.image(cv2_img, channels="BGR", caption="KNN Processed Image")
                else:
                    st.warning("No face detected. Please try again.")
        else:
            st.warning("No registered faces found. Please go to the 'Register New Face' tab.")
    
    else:
        DeepFace = load_deepface()
        if DeepFace is None:
            st.error("DeepFace not installed. Run: `pip install deepface tf-keras`")
        else:
            registered = get_registered_arcface_names()
            if len(registered) == 0:
                st.warning("No faces registered. Go to 'Register New Face' tab and use ArcFace mode.")
            else:
                st.caption(f"{len(registered)} face(s) registered in ArcFace")
                img_file_buffer = st.camera_input("Take a photo to mark attendance", key="attendance_cam_arc")
                
                if img_file_buffer is not None:
                    bytes_data = img_file_buffer.getvalue()
                    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    faces_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
                    det_faces = faces_detect.detectMultiScale(gray, 1.3, 5)
                    
                    if len(det_faces) == 0:
                        st.warning("⚠️ No face detected in the photo. Please try again.")
                    else:
                        with st.spinner("🔍 Running ArcFace recognition..."):
                            try:
                                results = DeepFace.find(
                                    img_path=cv2_img,
                                    db_path=ARCFACE_DB_DIR,
                                    model_name="ArcFace",
                                    detector_backend="opencv",
                                    distance_metric="cosine",
                                    enforce_detection=False,
                                    silent=True,
                                )
                                
                                matched = False
                                if len(results) > 0 and len(results[0]) > 0:
                                    match_df = results[0]
                                    match_df = match_df[match_df['distance'] < 0.40]
                                    
                                    if len(match_df) > 0:
                                        best = match_df.iloc[0]
                                        name = os.path.splitext(os.path.basename(best['identity']))[0].replace("_", " ")
                                        confidence = round((1 - best['distance']) * 100, 1)
                                        
                                        for (x,y,w,h) in det_faces:
                                            cv2.rectangle(cv2_img, (x,y), (x+w,y+h), (0,200,0), 2)
                                            cv2.putText(cv2_img, f"{name} ({confidence}%)", (x,y-15), 
                                                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), 2)
                                        
                                        ts = time.time()
                                        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                                        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
                                        
                                        try:
                                            attendance_data = pd.DataFrame([[name, timestamp, date]], columns=['Name', 'Time', 'Date'])
                                            attendance_data.to_sql('attendance', engine, if_exists='append', index=False)
                                            st.success(f"✅ Attendance marked for **{name}** — {confidence}% confidence (ArcFace)")
                                        except Exception as e:
                                            st.error(f"Error saving attendance: {e}")
                                        matched = True
                                
                                if not matched:
                                    for (x,y,w,h) in det_faces:
                                        cv2.rectangle(cv2_img, (x,y), (x+w,y+h), (0,0,255), 2)
                                        cv2.putText(cv2_img, "Unknown", (x,y-15), 
                                                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)
                                    st.warning("❌ Unknown person — no match found.")
                                
                                st.image(cv2_img, channels="BGR", caption="ArcFace Processed Image")
                            
                            except Exception as e:
                                st.error(f"ArcFace error: {e}")