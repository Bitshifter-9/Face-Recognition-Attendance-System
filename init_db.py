import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get('DATABASE_URL')

if not DATABASE_URL:
    print("Error: DATABASE_URL not set.")
    exit(1)

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    print("Creating 'registered_faces' table...")
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS registered_faces (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            face_encoding BYTEA NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    
    print("Ensuring 'attendance' table exists...")
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS attendance (
            "Name" TEXT,
            "Time" TEXT,
            "Date" TEXT
        );
    """))
    
    conn.commit()
    print("✅ Database tables created/verified.")
