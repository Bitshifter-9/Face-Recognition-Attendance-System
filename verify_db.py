
import os
import psycopg2
from sqlalchemy import create_engine, text

# Get URL from env or use the one provided by user
DATABASE_URL = os.environ.get('DATABASE_URL')

if not DATABASE_URL:
    print("Error: DATABASE_URL environment variable is not set.")
    exit(1)

print(f"Testing connection to: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else '...'} ...")

try:
    # Test with psycopg2 directly first
    conn = psycopg2.connect(DATABASE_URL)
    print("✅ Success! Connected to database using psycopg2.")
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    db_version = cursor.fetchone()
    print(f"   Database Version: {db_version[0]}")
    cursor.close()
    conn.close()

    # Test with SQLAlchemy (used in app)
    engine = create_engine(DATABASE_URL)
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print("✅ Success! SQLAlchemy engine is working.")
        
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    exit(1)
