# 🏋️ AI Virtual Fitness Coach

*Full Stack AI Fitness Application • Python • Streamlit • MediaPipe • Firebase*

## 🚀 Overview

**AI Virtual Fitness Coach** is a full-stack, AI-powered workout assistant that uses **real-time pose detection**, **rep counting**, **voice feedback**, and **cloud-based progress tracking**. Built using **Python**, **Streamlit**, **MediaPipe**, and **Firebase**, it turns your webcam into a smart fitness trainer.

You can perform 7 different exercises, track your reps/sets, receive voice guidance, and store your entire workout history securely in the cloud.

**Live Demo:** *Add your Streamlit Cloud link here*

---

## ✨ Features

### 🔹 **7 Supported Exercises**

* Bicep Curls
* Squats
* Push-ups
* Overhead Shoulder Press
* Lunges
* Jumping Jacks
* High Knees

### 🔹 **Real-Time Pose Detection**

Powered by **MediaPipe Pose**, the app tracks **33 human body keypoints** for accurate motion detection.

### 🔹 **Repetition & Set Counter**

Automatically counts:

* Reps
* Sets
* Exercise duration
* Workout summary

### 🔹 **Live Voice Feedback**

* Supports **Hindi/English**
* **Male/Female voices**
* Motivational prompts: “Rep complete!”, “Go lower!”, “Keep your posture straight!”

### 🔹 **User Authentication (Firebase)**

Secure **Email/Password Login & Signup** using Firebase Authentication.

### 🔹 **Workout History & Cloud Sync**

Every workout is saved to **Firebase Firestore** permanently:

* Exercise performed
* Sets completed
* Total reps
* Total time

Users can log in from any device and access their history.

### 🔹 **Advanced Analytics Dashboard (New)**

Get a complete performance report with:

* Total sessions, reps, and total workout time
* Daily consistency streak
* Exercise-wise breakdown table
* Daily reps trend chart
* One-click CSV export for report submission

### 🔹 **AI Training Planner (New)**

Built-in adaptive planner recommends your next targets based on:

* Goal type (**Hypertrophy / Strength / Endurance**)
* Current training load
* Intensity trend from your history

You can apply recommended reps/sets instantly.

### 🔹 **AI Coach + Smart Weekly Program (New)**

The app now includes an in-built AI coaching layer:

* AI-generated performance insights from your own workout history
* Smart weekly training plan generator
* Interactive “Ask AI Coach” assistant for form, recovery, progression, and nutrition basics
* Optional challenge mode for daily progression prompts
* **Real LLM support** via OpenAI API for context-aware, non-template answers

### 🔹 **Improved UI/UX (New)**

* Modern dark-gradient interface
* Card-based analytics and coaching sections
* Better information hierarchy for project demo presentation

### 🔹 **Body Transformation + Smart Nutrition System (New)**

Now supports complete profile-based planning:

* Body type tracking (mota/high body-fat, patla/lean, average)
* Goal modes: fat loss, muscle gain, recomposition
* Daily calories + macros calculator (BMR/TDEE based)
* Diet preference customization: Vegetarian / Vegan / Non-vegetarian
* Experience-level blueprint: Beginner / Intermediate / Pro athlete
* Voice summary for nutrition + plan recommendations

---

## 🛠️ Tools & Libraries Used

* **Python** – Primary programming language
* **Streamlit** – Frontend UI
* **OpenCV** – Webcam capture & image processing
* **MediaPipe** – Pose estimation model
* **NumPy** – Angle math & calculations
* **Pyrebase4** – Firebase authentication handling
* **Requests** – Firestore database communication

---

## 🚀 How to Run (Locally)

### **1️⃣ Clone the Repository**

```
git clone <your_repository_link>
cd ai-fitness-coach
```

### **2️⃣ Create & Activate Virtual Environment**

```
python -m venv venv
venv\Scripts\activate  # Windows
```

### **3️⃣ Install Required Libraries**

```
pip install -r requirements.txt
```

If you want real AI responses, set:

```
OPENAI_API_KEY=your_api_key_here
```

### **4️⃣ Setup Firebase**

1. Go to Firebase Console → Create Project
2. Enable Email/Password Login
3. Create Firestore Database → Start in Test Mode
4. Go to **Project Settings** → create a **Web App** (</>)
5. Copy **firebaseConfig** JSON keys
6. Manually add this key:

```
"databaseURL": "https://<your-project-id>.firebaseio.com/"
```

### **5️⃣ Run the App**

```
streamlit run app.py
```

### **6️⃣ Paste Firebase Config in App**

* Browser will open automatically
* Paste your **full Firebase JSON** in the sidebar
* When connection succeeds → "Firebase Connected!" will appear
* Now you can Sign Up / Log In

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push these 3 files to a new **Public GitHub Repo**:

   * `app.py`
   * `requirements.txt`
   * `utils.py`

2. Log in to **share.streamlit.io**

3. Click **New App** → Select your repository

4. Click **Advanced settings...**

5. In **Secrets**, paste your Firebase keys in TOML format:

```
FIREBASE_CONFIG = """
{
  "apiKey": "YOUR_KEY",
  "authDomain": "YOUR_DOMAIN",
  "projectId": "YOUR_PROJECT_ID",
  "storageBucket": "YOUR_BUCKET",
  "messagingSenderId": "XXXXXXX",
  "appId": "YOUR_APP_ID",
  "databaseURL": "YOUR_URL"
}
"""
```

6. Click **Deploy!**

Your AI Fitness Coach will go live on Streamlit Cloud.

---

## 🌟 Support

If you find this project helpful:
**⭐ Star the repository** and share it!
