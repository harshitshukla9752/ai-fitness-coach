# 🏋️ AI Virtual Fitness Coach

*Full Stack AI Fitness Application • Python • Streamlit • MediaPipe • Supabase*

## 🚀 Overview

**AI Virtual Fitness Coach** is a full-stack, AI-powered workout assistant that uses **real-time pose detection**, **rep counting**, **voice feedback**, and **cloud-based progress tracking**. Built using **Python**, **Streamlit**, **MediaPipe**, and **Supabase**, it turns your webcam into a smart fitness trainer.

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

### 🔹 **User Authentication (Supabase)**

Secure **Email/Password Login & Signup** using Supabase Authentication.

### 🔹 **Workout History & Cloud Sync**

Every workout is saved to **Supabase Postgres** permanently:

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
* **Supabase** – Authentication and database storage

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

For Gemini (optional alternative), you can similarly use:

```
GEMINI_API_KEY=your_key_here
```

> API keys can be generated from:
> - OpenAI dashboard: https://platform.openai.com/api-keys
> - Google AI Studio (Gemini): https://aistudio.google.com/app/apikey

### **4️⃣ Setup Supabase Auth + DB**

This app uses **Supabase only** for login/signup and workout/profile storage. Add credentials in `.streamlit/secrets.toml`:

```toml
[supabase]
url = "https://<project-ref>.supabase.co"
key = "<anon-public-key>"
```

Then create the database tables, indexes, triggers, and RLS policies by copying and running `supabase/schema.sql` in the Supabase SQL Editor. Detailed guide is available in `supabase/README.md`.

Detailed guide is available in `supabase/README.md`.

### **5️⃣ Run the App**

```
streamlit run app.py
```

Open browser: `http://localhost:8501`

### **6️⃣ Save User Profile Once (No Repeated Questions)**

After login, go to **AI Training Planner** tab and click **Save Profile for Next Login**.  
Your body type, goals, diet preference, and activity profile will auto-load from Supabase on future logins.

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push these 3 files to a new **Public GitHub Repo**:

   * `app.py`
   * `requirements.txt`
   * `utils.py`

2. Log in to **share.streamlit.io**

3. Click **New App** → Select your repository

4. Click **Advanced settings...**

5. In **Secrets**, paste your Supabase keys in TOML format:

```toml
[supabase]
url = "https://<project-ref>.supabase.co"
key = "<anon-public-key>"
```

6. Run `supabase/schema.sql` in Supabase SQL Editor once, then click **Deploy!**

Your AI Fitness Coach will go live on Streamlit Cloud.

### Netlify note

Netlify static hosting directly Streamlit apps ko run nahi karta.  
Recommended deployment options for this project:

1. **Streamlit Community Cloud** (easiest)
2. **Render / Railway / VPS + Docker**
3. Netlify only as static landing page + external hosted Streamlit app link

---

## 🌟 Support

If you find this project helpful:
**⭐ Star the repository** and share it!
