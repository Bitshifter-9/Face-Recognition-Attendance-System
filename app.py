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

st.set_page_config(page_title="Attendance System", layout="wide")
st.title("Face Recognition Attendance System")