# Face Recognition Attendance System

This project automatically takes attendance using face recognition. Pretty cool right? Just look at the camera and it recognizes you and marks your attendance.

I've included a demo video (Demo.mp4) - check it out to see how it works!
[![Watch the demo]](demo.mp4)

## What does it do?

Basically, the system learns faces from your webcam, then whenever it sees someone it knows, it can mark their attendance. The attendance gets saved in CSV files with timestamps, so you can see who was present and when. I also added a simple Streamlit dashboard to view the records.

## What you need

- Python (I used 3.8+)
- A working webcam
- That's pretty much it

## Getting Started

First, clone this repo and install the dependencies:

```bash
git clone https://github.com/yourusername/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System
pip install -r requirements.txt
```

## How to use it

**1. Register faces first**

Run this to add people to the system:

```bash
python add_faces.py
```

It'll ask for your name, then just look at the camera. It captures 100 images of your face from different angles. Press 'q' when it's done. Do this for everyone you want to register.

**Important:** Make sure you let it capture all 100 images for each person - the KNN algorithm needs this many samples to accurately recognize faces. Less than that and the accuracy drops significantly.

**2. Start taking attendance**

Now run the main script:

```bash
python test.py
```

Your webcam will open and start recognizing faces. When it detects someone, their name pops up on the screen. Press 'O' to mark their attendance - it gets saved to a CSV file in the `attendance` folder with the date. Press 'q' to quit.

**3. Check the attendance**

To view attendance records in a nicer format:

```bash
streamlit run app.py
```

This opens a web page where you can see the attendance data. 

**Note:** Sometimes Streamlit can be a bit slow to refresh, so if you don't see the latest attendance right away, just check the CSV files directly in the `attendance` folder.

## Files in this project

```
add_faces.py          # Adds new people to the system
test.py               # Main attendance program  
app.py                # Web dashboard
requirements.txt      # All the libraries you need
Demo.mov             # Demo video
data/                # Face data gets stored here
attendance/          # Attendance CSV files go here
```

## How it works

The system uses OpenCV to detect faces with Haar Cascades, then trains a KNN classifier on the captured images. When you run it, it compares detected faces against the trained model and identifies people. Pretty straightforward!

## Tech Stack

- OpenCV - for face detection
- scikit-learn - KNN classifier
- NumPy - handling arrays
- Pandas & Streamlit - for the dashboard

Feel free to fork this or suggest improvements!
