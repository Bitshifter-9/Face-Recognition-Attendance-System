# Face Recognition Attendance System

A web-based attendance system using Face Recognition, Streamlit, and PostgreSQL.

## � How to Use
1.  **Register New Face**:  
    -   Go to the **"Register New Face"** tab.
    -   Enter your name and look at the camera.
    -   Click **"Save Face"** and wait for it to generate 100 samples.
2.  **Mark Attendance**:  
    -   Switch to the **"Mark Attendance"** tab.
    -   Stand in front of the camera.
    -   It will recognize you and mark you present automatically.
3.  **View Records**:  
    -   Check the **"View Records"** tab to see the attendance logs.

## �🚀 Features
-   **Web Interface**: Access via browser on any device.
-   **Cloud Database**: Secure PostgreSQL storage for faces and logs.
-   **Smart Registration**: Auto-generates 100 face samples for better accuracy.
-   **Instant Tracking**: Real-time recognition and timestamping.
-   **Dashboard**: Manage records easily.

## 🛠️ Tech Stack
-   **Frontend**: Streamlit
-   **Vision**: OpenCV, face_recognition
-   **Database**: PostgreSQL
-   **ML**: Scikit-learn (KNN)

## 💻 Local Setup
1.  **Clone**:
    ```bash
    git clone https://github.com/Bitshifter-9/Face-Recognition-Attendance-System.git
    cd Face-Recognition-Attendance-System
    ```
2.  **Install**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Config**:
    Set `DATABASE_URL` (e.g., local or cloud Postgres).
    ```bash
    export DATABASE_URL="postgresql://user:pass@host:5432/db" # Mac/Linux
    $env:DATABASE_URL="postgresql://..." # Windows
    ```
4.  **Run**:
    ```bash
    streamlit run app.py
    ```

## 📂 Structure
-   `app.py`: Main app (Register, Mark, View).
-   `requirements.txt`: Python deps.
-   `packages.txt`: System deps.
-   `init_db.py`: DB setup script.
