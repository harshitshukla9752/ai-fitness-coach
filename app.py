import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from utils import calculate_angle # Backend helper function ko import karein
import streamlit.components.v1 as components # Voice Assistant ke liye
import pyrebase # Pyrebase use karein
import json
import requests # Naya tareeka: REST API ke liye

# --- Voice Assistant (TTS) Function ---
def speak(text, lang, voice_name):
    # (Is code mein koi change nahi)
    text = text.replace("'", "").replace('"', '')
    speech_js = f"""
        <script>
            const text = "{text}";
            const lang = "{lang}";
            const voiceName = "{voice_name}";
            function doSpeak() {{
                const utter = new SpeechSynthesisUtterance(text);
                utter.lang = lang;
                if (voiceName) {{
                    const voices = window.speechSynthesis.getVoices();
                    const selectedVoice = voices.find(v => v.name === voiceName);
                    if (selectedVoice) {{
                        utter.voice = selectedVoice;
                    }} else {{
                        console.warn(`Voice '{{voiceName}}' not found.`);
                    }}
                }}
                window.speechSynthesis.speak(utter);
            }}
            if (window.speechSynthesis.getVoices().length > 0) {{ doSpeak(); }}
            else {{ window.speechSynthesis.onvoiceschanged = doSpeak; }}
        </script>
    """
    components.html(speech_js, height=0, width=0)

# --- MediaPipe ko initialize karein (Backend) ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Streamlit Session State (App ki Memory) ---
default_states = {
    'voice_enabled': True, 'voice_lang': 'hi-IN', 'voice_name': 'Google हिन्दी',
    'workout_log': [], 
    'rep_counter_left': 0, 'rep_counter_right': 0, 'current_set': 1, # NAYA: Set counter
    'target_reps': 10, 'target_sets': 3, # NAYA: Target goals
    'stage_left': 'down', 'stage_right': 'down', 'stage': 'down',
    'feedback': 'Start your workout!', 'last_spoken_feedback': '',
    'start_time': 0, 'webcam_started': False,
    'firebase_config': None, 'firebase': None, # Config ke liye naya logic
    'auth': None, 'user': None, 'db': None, 'page': 'Login',
    'workout_complete_feedback_given': False 
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper functions (Speak, Reset, TTS) ---
def safe_speak(text):
    if st.session_state.voice_enabled:
        speak(text, st.session_state.voice_lang, st.session_state.voice_name)

def reset_states(exercise_choice):
    st.session_state.rep_counter_left = 0
    st.session_state.rep_counter_right = 0
    st.session_state.current_set = 1
    st.session_state.workout_complete_feedback_given = False
    
    if exercise_choice in ['Squats', 'Push-ups', 'Lunges', 'Jumping Jacks', 'High Knees']:
        initial_stage = 'up'
    else: 
        initial_stage = 'down'
        
    st.session_state.stage_left = initial_stage
    st.session_state.stage_right = initial_stage
    st.session_state.stage = initial_stage
    
    st.session_state.feedback = f'Starting Set 1 of {st.session_state.target_sets}. Get in position!'
    st.session_state.last_spoken_feedback = st.session_state.feedback
    safe_speak(st.session_state.feedback)
    st.session_state.start_time = time.time()

VOICE_OPTIONS = {
    'hi-IN': {
        'Hindi (Male - Default)': 'Google हिन्दी',
        'Hindi (Female - Realistic)': 'Microsoft Kalpana - Hindi (India)'
    },
    'en-US': {
        'English (Female - Default)': 'Google US English',
        'English (Female - Realistic)': 'Microsoft Zira - English (United States)',
        'English (Male - Realistic)': 'Microsoft David - English (UnitedStates)'
    }
}

# -----------------------------------------------------------------
# --- DATABASE LOGIC (REST API - Yahi code hai, koi change nahi) ---
# -----------------------------------------------------------------

def get_db_url_base(config):
    project_id = config.get("projectId")
    return f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents"

def get_headers(user_token):
    return {
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/json"
    }

def save_workout_log_rest(config, user_token, workout_data):
    """Workout log ko REST API se Firestore mein save karein."""
    try:
        user_id = st.session_state.user['localId']
        collection_path = f"user_logs_{user_id}"
        base_url = get_db_url_base(config)
        url = f"{base_url}/{collection_path}"
        headers = get_headers(user_token)
        
        firestore_document = {
            "fields": {
                "exercise": {"stringValue": workout_data["exercise"]},
                "side": {"stringValue": workout_data["side"]},
                "sets_completed": {"integerValue": workout_data["sets_completed"]},
                "reps_per_set": {"integerValue": workout_data["reps_per_set"]},
                "duration": {"doubleValue": workout_data["duration"]},
                "timestamp": {"timestampValue": f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"}
            }
        }
        
        response = requests.post(url, json=firestore_document, headers=headers)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Database save error: {e}")
        return False

def load_workout_logs_rest(config, user_token):
    """Workout logs ko REST API se load karein."""
    try:
        user_id = st.session_state.user['localId']
        collection_path = f"user_logs_{user_id}"
        base_url = get_db_url_base(config)
        url = f"{base_url}/{collection_path}?orderBy=timestamp desc"
        headers = get_headers(user_token)
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        logs = []
        
        if "documents" in data:
            for doc in data["documents"]:
                fields = doc.get("fields", {})
                logs.append({
                    "exercise": fields.get("exercise", {}).get("stringValue", "N/A"),
                    "side": fields.get("side", {}).get("stringValue", "N/A"),
                    "sets_completed": int(fields.get("sets_completed", {}).get("integerValue", 0)),
                    "reps_per_set": int(fields.get("reps_per_set", {}).get("integerValue", 0)),
                    "duration": float(fields.get("duration", {}).get("doubleValue", 0.0)),
                })
        return logs
    except Exception as e:
        return []

# -----------------------------------------------------------------
# --- END OF DATABASE LOGIC ---
# -----------------------------------------------------------------


# --- Main App ---
st.title("🏋️ AI Virtual Fitness Coach")

# --- 1. NAYA: Firebase Config Logic (Secrets se padhne ke liye) ---
if 'firebase_config' not in st.session_state or st.session_state.firebase_config is None:
    # Pehle Streamlit ke server "Secrets" se check karein
    if 'firebase_config' in st.secrets:
        config = dict(st.secrets.firebase_config) # Secrets se load karein
        st.session_state.firebase_config = config
        st.sidebar.success("Firebase Config Loaded from Secrets!")
    else:
        # Agar Secrets mein nahi mila, toh sidebar mein box dikhayein (local testing ke liye)
        st.sidebar.title("Configuration")
        st.sidebar.info("Apna Firebase project config yahaan paste karein.")
        config_input = st.sidebar.text_area("Firebase Config (JSON format)", 
                                            height=300, 
                                            key="firebase_config_input_widget")
        if config_input:
            try:
                config = json.loads(config_input)
                st.session_state.firebase_config = config
                st.sidebar.success("Firebase Config Loaded! Rerunning...")
                time.sleep(1)
                st.rerun() # Rerun taaki config load ho jaaye
            except Exception as e:
                st.sidebar.error(f"Error: Ye JSON format sahi nahi hai. {e}")
                st.stop() # Config galat hai, aage na badhein
        else:
            st.warning("Please sidebar mein Firebase config paste karein.")
            st.stop() # Config nahi hai, aage na badhein

# --- 2. Initialize Firebase (Ab config 100% hai) ---
if 'firebase' not in st.session_state or st.session_state.firebase is None:
    try:
        config = st.session_state.firebase_config
        firebase = pyrebase.initialize_app(config)
        st.session_state.firebase = firebase
        st.session_state.auth = firebase.auth()
    except Exception as e:
        st.error(f"Firebase initialize nahi hua. Config check karein. Error: {e}")
        st.stop()


# Page routing logic
if st.session_state.page == 'Login' and st.session_state.user:
    st.session_state.page = 'Coach'
if st.session_state.page == 'Coach' and not st.session_state.user:
    st.session_state.page = 'Login'

# --- 3. Login / Signup Page ---
if st.session_state.page == 'Login':
    st.header("Login / Sign Up")
    
    choice = st.radio("Chunein:", ("Login", "Sign Up"))
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if choice == "Sign Up":
        if st.button("Sign Up"):
            try:
                user = st.session_state.auth.create_user_with_email_and_password(email, password)
                st.session_state.user = user
                st.success("Account ban gaya! Login ho raha hai...")
                safe_speak("Account created! Logging you in.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Signup Error: {e}")

    if choice == "Login":
        if st.button("Login"):
            try:
                user = st.session_state.auth.sign_in_with_email_and_password(email, password)
                st.session_state.user = user
                
                # NAYA LOGIC: Workout logs ko REST API se load karein
                token = st.session_state.user['idToken']
                config = st.session_state.firebase_config
                st.session_state.workout_log = load_workout_logs_rest(config, token)
                
                st.success("Login successful!")
                safe_speak("Login successful!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Login Error: {e}")

# --- 4. Main Coach Page (Login ke baad) ---
elif st.session_state.page == 'Coach':
    
    # --- Sidebar (Coach Settings) ---
    st.sidebar.title(f"Welcome, {st.session_state.user['email']}")
    
    exercise_choice = st.sidebar.selectbox(
        "Exercise chunein:",
        ("Bicep Curls", "Squats", "Push-ups", "Overhead Press", "Lunges", "Jumping Jacks", "High Knees"),
        key="exercise_choice"
    )

    if exercise_choice in ["Bicep Curls", "Push-ups", "Overhead Press"]:
        side_label = "Kaunsa haath track karein?"
    elif exercise_choice in ["Squats", "Lunges", "High Knees"]:
        side_label = "Kaunsa pair track karein?"
    else: 
        side_label = "Tracking:"
    
    side_options = ("Left", "Right", "Both")
    if exercise_choice in ["Jumping Jacks"]:
        side_options = ("Both",)
        
    side_choice = st.sidebar.selectbox(
        side_label,
        side_options, 
        key="side_choice"
    )
    
    st.sidebar.divider()

    # --- Target Goals ---
    st.sidebar.title("🎯 Your Targets")
    st.session_state.target_reps = st.sidebar.number_input("Target Reps (per set)", min_value=1, value=st.session_state.target_reps)
    st.session_state.target_sets = st.sidebar.number_input("Target Sets", min_value=1, value=st.session_state.target_sets)

    st.sidebar.divider()
    
    # --- Voice Assistant ---
    st.sidebar.title("🗣️ Voice Assistant")
    st.session_state.voice_enabled = st.sidebar.checkbox("Enable Voice Assistant", value=st.session_state.voice_enabled)
    lang_map = {'Hindi': 'hi-IN', 'English': 'en-US'}
    default_lang_index = 0 if st.session_state.voice_lang == 'hi-IN' else 1
    selected_lang_friendly = st.sidebar.selectbox("Assistant Language:", ('Hindi', 'English'), index=default_lang_index)
    st.session_state.voice_lang = lang_map[selected_lang_friendly]
    
    available_voices = VOICE_OPTIONS[st.session_state.voice_lang]
    current_voice_name = st.session_state.voice_name
    current_voice_friendly_name = None
    for friendly_name, internal_name in available_voices.items():
        if internal_name == current_voice_name:
            current_voice_friendly_name = friendly_name
            break
    default_voice_index = 0
    if current_voice_friendly_name:
        default_voice_index = list(available_voices.keys()).index(current_voice_friendly_name)
    selected_voice_friendly = st.sidebar.selectbox("Assistant Voice (Male/Female):", available_voices.keys(), index=default_voice_index)
    st.session_state.voice_name = available_voices[selected_voice_friendly]
    
    st.sidebar.divider()
    
    # --- Workout Log (Sidebar) ---
    st.sidebar.title("📊 Your Workout Log")
    if not st.session_state.workout_log:
        st.sidebar.info("Aapka koi workout log nahi hai.")
    else:
        for i, log in enumerate(st.session_state.workout_log):
            st.sidebar.success(
                f"**Session {len(st.session_state.workout_log) - i}:** {log['exercise']} ({log['side']})\n"
                f"{log['sets_completed']} Sets x {log['reps_per_set']} Reps\n"
                f"Duration: {log['duration']:.0f}s"
            )
            
    # --- Logout Button (Sidebar) ---
    if st.sidebar.button("Logout", use_container_width=True, type="secondary"):
        st.session_state.user = None
        st.session_state.auth = None
        st.session_state.workout_log = []
        st.success("Logged out successfully!")
        safe_speak("Logged out successfully!")
        time.sleep(1)
        st.rerun()

    # --- Main App Interface (Coach) ---
    st.caption(f"Aapne chuna hai: **{exercise_choice} ({side_choice})**.")
    
    if st.button("Start / Stop Webcam", key="start_stop_button", use_container_width=True, type="primary"):
        st.session_state.webcam_started = not st.session_state.webcam_started
        
        if st.session_state.webcam_started:
            reset_states(exercise_choice)
        
        elif not st.session_state.webcam_started and st.session_state.start_time != 0:
            # Stop logic
            final_sets = st.session_state.current_set
            # Agar aakhri set poora nahi hua, toh use count na karein
            if max(st.session_state.rep_counter_left, st.session_state.rep_counter_right) < st.session_state.target_reps:
                final_sets -= 1
                
            final_reps_per_set = st.session_state.target_reps
            final_duration = time.time() - st.session_state.start_time
            
            if final_sets > 0:
                log_data = {
                    "exercise": exercise_choice,
                    "side": side_choice,
                    "sets_completed": final_sets,
                    "reps_per_set": final_reps_per_set,
                    "duration": final_duration
                }
                
                token = st.session_state.user['idToken']
                config = st.session_state.firebase_config
                save_success = save_workout_log_rest(config, token, log_data)
                
                if save_success:
                    st.session_state.workout_log.insert(0, log_data)
                    log_text = f"Workout logged! {final_sets} sets of {final_reps_per_set} reps."
                    st.success(log_text)
                    safe_speak(log_text)
                else:
                    st.error("Workout log save nahi hua (Database Error).")
                    safe_speak("Failed to log workout.")
            else:
                st.warning("Koi set complete nahi hua. Workout log nahi hua.")
                safe_speak("No sets completed. Workout not logged.")
            
            st.session_state.start_time = 0
            st.rerun()

    # --- Stats aur Video ke liye Placeholders (Frontend UI) ---
    stats_placeholder = st.empty()
    video_placeholder = st.empty()

    # --- Main Backend Loop ---
    if st.session_state.webcam_started:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam nahi chala. Permissions check karein.")
        else:
            while st.session_state.webcam_started:
                ret, frame = cap.read()
                if not ret: break

                elapsed_time = time.time() - st.session_state.start_time
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                angle_l, angle_r = 0, 0
                feedback_msg = st.session_state.feedback
                
                workout_finished = st.session_state.current_set > st.session_state.target_sets
                if workout_finished and not st.session_state.workout_complete_feedback_given:
                    feedback_msg = "Workout Complete! Stop the camera to save."
                    st.session_state.workout_complete_feedback_given = True
                
                try:
                    if results.pose_landmarks and not workout_finished:
                        landmarks = results.pose_landmarks.landmark
                        
                        # (Exercise logic mein koi change nahi)
                        if exercise_choice in ["Bicep Curls", "Push-ups", "Overhead Press"]:
                            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                            angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                        elif exercise_choice in ["Squats", "Lunges", "High Knees"]:
                            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                            ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                            angle_l = calculate_angle(hip_l, knee_l, ankle_l)
                            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                            angle_r = calculate_angle(hip_r, knee_r, ankle_r)
                        elif exercise_choice == "Jumping Jacks":
                            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                            angle_l = calculate_angle(hip_l, shoulder_l, wrist_l)
                            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                            angle_r = calculate_angle(hip_r, shoulder_r, wrist_r)

                        # (Threshold logic mein koi change nahi)
                        if exercise_choice == "Bicep Curls":
                            up_threshold, down_threshold, stage_check = 160, 30, 'down'
                        elif exercise_choice == "Squats":
                            up_threshold, down_threshold, stage_check = 160, 90, 'up'
                        elif exercise_choice == "Push-ups":
                            up_threshold, down_threshold, stage_check = 160, 90, 'up'
                        elif exercise_choice == "Overhead Press":
                            up_threshold, down_threshold, stage_check = 160, 90, 'down'
                        elif exercise_choice == "Lunges":
                            up_threshold, down_threshold, stage_check = 160, 100, 'up'
                        elif exercise_choice == "Jumping Jacks":
                            up_threshold, down_threshold, stage_check = 160, 30, 'up'
                        elif exercise_choice == "High Knees":
                            up_threshold, down_threshold, stage_check = 160, 90, 'up'

                        # Rep Counting Logic (Target ke saath)
                        def check_set_completion(feedback_msg):
                            current_reps = max(st.session_state.rep_counter_left, st.session_state.rep_counter_right)
                            if current_reps >= st.session_state.target_reps:
                                st.session_state.current_set += 1
                                if st.session_state.current_set <= st.session_state.target_sets:
                                    feedback_msg = f"Set {st.session_state.current_set - 1} complete! Next set: Get ready."
                                    st.session_state.rep_counter_left = 0
                                    st.session_state.rep_counter_right = 0
                                    st.session_state.stage_left = stage_check
                                    st.session_state.stage_right = stage_check
                                    st.session_state.stage = stage_check
                                else:
                                    feedback_msg = "Workout Complete! Stop the camera to save."
                            return feedback_msg

                        if side_choice == 'Left':
                            if angle_l < down_threshold and st.session_state.stage_left == stage_check:
                                st.session_state.stage_left = 'down' if stage_check == 'up' else 'up'
                                st.session_state.rep_counter_left += 1
                                feedback_msg = f'Rep {st.session_state.rep_counter_left}!'
                                feedback_msg = check_set_completion(feedback_msg)
                            elif angle_l > up_threshold and st.session_state.stage_left != stage_check:
                                st.session_state.stage_left = stage_check
                                if feedback_msg != "Workout Complete! Stop the camera to save.": # Taaki complete msg overwrite na ho
                                    feedback_msg = f'Set {st.session_state.current_set}: Ready'

                        elif side_choice == 'Right':
                            if angle_r < down_threshold and st.session_state.stage_right == stage_check:
                                st.session_state.stage_right = 'down' if stage_check == 'up' else 'up'
                                st.session_state.rep_counter_right += 1
                                feedback_msg = f'Rep {st.session_state.rep_counter_right}!'
                                feedback_msg = check_set_completion(feedback_msg)
                            elif angle_r > up_threshold and st.session_state.stage_right != stage_check:
                                st.session_state.stage_right = stage_check
                                if feedback_msg != "Workout Complete! Stop the camera to save.":
                                    feedback_msg = f'Set {st.session_state.current_set}: Ready'
                        
                        elif side_choice == 'Both':
                            stage_l_reached = angle_l < down_threshold
                            stage_r_reached = angle_r < down_threshold
                            stage_l_reset = angle_l > up_threshold
                            stage_r_reset = angle_r > up_threshold

                            if (stage_l_reached and stage_r_reached and st.session_state.stage == stage_check):
                                st.session_state.stage = 'down' if stage_check == 'up' else 'up'
                                st.session_state.rep_counter_left += 1
                                st.session_state.rep_counter_right += 1
                                feedback_msg = f'Rep {st.session_state.rep_counter_left}!'
                                feedback_msg = check_set_completion(feedback_msg)
                            elif (stage_l_reset and stage_r_reset and st.session_state.stage != stage_check):
                                st.session_state.stage = stage_check
                                if feedback_msg != "Workout Complete! Stop the camera to save.":
                                    feedback_msg = f'Set {st.session_state.current_set}: Ready'
                            elif (stage_l_reached and not stage_r_reached and st.session_state.stage == stage_check):
                                feedback_msg = 'ERROR: Move right side too!'
                            elif (not stage_l_reached and stage_r_reached and st.session_state.stage == stage_check):
                                feedback_msg = 'ERROR: Move left side too!'
                            
                            if exercise_choice in ['Squats', 'Push-ups', 'Lunges']:
                                if (stage_check == 'up' and 
                                    (down_threshold < angle_l < up_threshold or down_threshold < angle_r < up_threshold) and
                                    st.session_state.stage == 'up'):
                                    feedback_msg = 'Go lower!'
                        
                        st.session_state.feedback = feedback_msg
                        mp_drawing.draw_landmarks(
                            image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                        )
                except Exception as e:
                    # Jumping jacks mein kabhi kabhi wrist nahi dikhti
                    if exercise_choice == "Jumping Jacks":
                        feedback_msg = "Keep arms in frame!"
                    else:
                        feedback_msg = "Poora shareer camera mein dikhayein!"
                    st.session_state.feedback = feedback_msg

                # Voice Assistant Logic
                if (st.session_state.voice_enabled and 
                    st.session_state.feedback != st.session_state.last_spoken_feedback):
                    st.session_state.last_spoken_feedback = st.session_state.feedback
                    safe_speak(st.session_state.feedback)

                # Frontend UI: Stats (Target ke saath)
                if side_choice == 'Left':
                    rep_display = st.session_state.rep_counter_left
                elif side_choice == 'Right':
                    rep_display = st.session_state.rep_counter_right
                else: 
                    rep_display = st.session_state.rep_counter_left
                
                if workout_finished:
                    set_display = "Done!"
                    rep_display = "Done!"
                else:
                    set_display = f"{st.session_state.current_set} / {st.session_state.target_sets}"
                    rep_display = f"{rep_display} / {st.session_state.target_reps}"

                stats_placeholder.markdown(f"""
                    <div style="background-color: #222; padding: 15px; border-radius: 10px; font-size: 1.5rem; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                        <div style="text-align: center;">
                            <strong>SET</strong><br><span style="color: #00FF00; font-size: 2.5rem;">{set_display}</span>
                        </div>
                        <div style="text-align: center;">
                            <strong>TIMER</strong><br><span style="color: #00FF00; font-size: 2.5rem;">{int(elapsed_time)}s</span>
                        </div>
                        <div style="text-align: center;">
                            <strong>REPS</strong><br><span style="color: #00FF00; font-size: 2.5rem;">{rep_display}</span>
                        </div>
                    </div>
                    <div style="font-size: 1.5rem; text-align: center; margin-top: 15px; color: #00FFFF;">
                        <strong>FEEDBACK:</strong> {st.session_state.feedback}
                    </div>
                """, unsafe_allow_html=True)
                
                video_placeholder.image(image_bgr, channels="BGR", width='stretch')
                
                if not st.session_state.webcam_started:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            stats_placeholder.empty()
            video_placeholder.empty()