🏋️ AI Virtual Fitness Coach

This is a full-stack AI-powered fitness coach built using Python, Streamlit, MediaPipe, and Firebase. It tracks your workouts in real-time, counts repetitions, and saves your progress permanently.

Live Demo Link: (Add your deployed Streamlit Cloud link here)

✨ Features

7 Different Exercises: Includes Bicep Curls, Squats, Push-ups, Overhead Press, Lunges, Jumping Jacks, and High Knees.

Real-time Pose Estimation: Utilizes Google's MediaPipe to track 33 different human body landmarks from a live webcam feed.

Rep & Set Counter: Tracks your entire workout based on user-defined Target Reps and Target Sets.

Live Voice Feedback: A multi-language (Hindi/English) and multi-voice (Male/Female) assistant provides real-time audible feedback like "Rep complete!" or "Go lower!"

User Authentication: Secure Login/Signup system powered by Firebase Authentication (Email/Password).

Persistent Workout Log: All completed workout sessions (exercise, sets, reps, and duration) are saved permanently to a Firebase Firestore database.

🛠️ Tools & Libraries Used

Python: The core programming language.

Streamlit: To build the interactive frontend web application.

OpenCV: To capture and process the live video feed from the webcam.

MediaPipe: For the real-time AI pose estimation model.

NumPy: For high-performance mathematical calculations (joint angles).

Pyrebase4: Python wrapper for Firebase Authentication (Login/Signup).

Requests: To communicate with the Firebase Firestore REST API for database operations (saving/loading logs).

🚀 How to Run (Locally)

Clone the Repository:

git clone (your repository link)
cd ai-fitness-coach


Create and Activate Virtual Environment:

python -m venv venv
venv\Scripts\activate


Install Required Libraries:
Use the provided requirements.txt file.

pip install -r requirements.txt


Get Firebase Keys:

Go to Firebase and create a new project.

Go to Authentication -> Sign-in method -> Enable Email/Password.

Go to Firestore Database -> Create database -> Start in Test mode.

Go to Project Settings -> Register a new Web App (</>) and copy the firebaseConfig keys.

Run the App:

streamlit run app.py


Paste your Firebase Config:

The app will open in your browser.

Paste the complete firebaseConfig JSON (including a manually added databaseURL key) into the sidebar text area.

A "Firebase Connected!" message will appear. You can now Sign up or Login.

☁️ Deployment (Streamlit Community Cloud)

Push your three project files (app.py, requirements.txt, utils.py) to a new Public GitHub Repository.

Log in to Streamlit Community Cloud with your GitHub account.

Click "New app" and select your repository.

Click on "Advanced settings...".

In the "Secrets" box, paste your firebaseConfig keys in TOML format (as shown in the previous deployment steps).

Click "Deploy!".
