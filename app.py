import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from utils import calculate_angle
import streamlit.components.v1 as components
import pyrebase
import json
import requests
import gc # Memory fix ke liye

# --- Voice Assistant (TTS) Function ---
def speak(text, lang, voice_name):
    # JavaScript mein quotes (') se problem ho sakti hai, isliye unhe hata dein
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
            // Voices load hone ka wait karein
            if (window.speechSynthesis.getVoices().length > 0) {{ doSpeak(); }}
            else {{ window.speechSynthesis.onvoiceschanged = doSpeak; }}
        </script>
    """
    components.html(speech_js, height=0, width=0)

# --- AI Models ko Cache Karein (Memory Fix) ---
@st.cache_resource
def load_models():
    mp_pose = mp.solutions.pose
    # 'model_complexity=0' (Lite model) use karein taaki server par crash na ho
    pose = mp_pose.Pose(
        model_complexity=0, # 0=lite, 1=full, 2=heavy
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    return mp_pose, pose, mp_drawing

# Models ko load karein (cached)
mp_pose, pose, mp_drawing = load_models()

# --- Session State Initialization ---
default_states = {
    'voice_enabled': True, 'voice_lang': 'hi-IN', 'voice_name': 'Google हिन्दी',
    'workout_log': [], 'rep_counter_left': 0, 'rep_counter_right': 0, 'set_counter': 1,
    'stage_left': 'down', 'stage_right': 'down', 'stage': 'down',
    'feedback': 'Start your workout!', 'last_spoken_feedback': '',
    'start_time': 0, 'webcam_started': False,
    'firebase_config_input': '', 'firebase_config': None, 'firebase': None,
    'auth': None, 'user': None, 'page': 'Login',
    'target_reps': 10, 'target_sets': 3
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
    # Set counter ko reset nahi karenge, taaki woh badhta rahe
    if exercise_choice in ['Squats', 'Push-ups', 'Lunges', 'Jumping Jacks', 'High Knees']:
        initial_stage = 'up'
    else: 
        initial_stage = 'down'
    st.session_state.stage_left = initial_stage
    st.session_state.stage_right = initial_stage
    st.session_state.stage = initial_stage
    st.session_state.feedback = f'Set {st.session_state.set_counter}! Get in position!'
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
        'English (Male - Realistic)': 'Microsoft David - English (United States)'
    }
}

# --- YAHI HAI ASLI FIX: Database Logic (REST API) ---
# Ismein .firestore() ka istemaal hi nahi hai
def get_db_url(config, user_token):
    project_id = config.get("projectId")
    user_id = st.session_state.user['localId']
    # Naya Fix: Har user ka data ek unique collection mein
    collection_path = f"user_logs_{user_id}" 
    base_url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/{collection_path}"
    return base_url, user_token

def save_workout_log_rest(config, user_token, workout_data):
    """Workout log ko REST API se Firestore mein save karein."""
    try:
        base_url, token = get_db_url(config, user_token)
        url = f"{base_url}" # Naye document ke liye collection URL
        
        # Naya Fix: Auth token ko URL se hata kar Header mein daalein
        headers = {
            "Authorization": f"Bearer {user_token}",
            "Content-Type": "application/json"
        }
        
        firestore_document = {
            "fields": {
                "exercise": {"stringValue": workout_data["exercise"]},
                "side": {"stringValue": workout_data["side"]},
                "reps_left": {"integerValue": workout_data["reps_left"]},
                "reps_right": {"integerValue": workout_data["reps_right"]},
                "duration": {"doubleValue": workout_data["duration"]},
                "set_number": {"integerValue": workout_data["set_number"]},
                "target_reps": {"integerValue": workout_data["target_reps"]},
                "timestamp": {"timestampValue": f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"}
            }
        }
        
        response = requests.post(url, json=firestore_document, headers=headers)
        response.raise_for_status() # Agar error ho toh ruk jaaye
        return True
    except Exception as e:
        # Error ko response se print karein
        error_details = e
        try:
            error_details = response.json()
        except:
            pass
        st.error(f"Database save error: {error_details}")
        return False

def load_workout_logs_rest(config, user_token):
    """Workout logs ko REST API se load karein."""
    try:
        base_url, token = get_db_url(config, user_token)
        url = f"{base_url}?orderBy=timestamp desc"
        
        # Naya Fix: Auth token ko Header mein daalein
        headers = {
            "Authorization": f"Bearer {user_token}",
            "Content-Type": "application/json"
        }
        
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
                    "reps_left": int(fields.get("reps_left", {}).get("integerValue", 0)),
                    "reps_right": int(fields.get("reps_right", {}).get("integerValue", 0)),
                    "duration": float(fields.get("duration", {}).get("doubleValue", 0.0)),
                    "set_number": int(fields.get("set_number", {}).get("integerValue", 1)),
                    "target_reps": int(fields.get("target_reps", {}).get("integerValue", 10)),
                })
        return logs
    except Exception as e:
        error_details = e
        try:
            error_details = response.json()
        except:
            pass
        st.error(f"Database load error: {error_details}")
        return []

# -----------------------------------------------------------------
# --- END OF NEW DATABASE LOGIC ---
# -----------------------------------------------------------------


# --- Main App ---
st.title("🏋️ AI Virtual Fitness Coach")

# --- Deployment Logic (Secrets / Sidebar) ---
firebase_config_json = None
# 1. Pehle 'Secrets' check karein (Server ke liye)
if 'firebase_config' in st.secrets:
    firebase_config_json = json.dumps(st.secrets.firebase_config)
    st.session_state.firebase_config_input = firebase_config_json # Save karein
# 2. Agar nahi mila, toh sidebar (Local test ke liye)
else:
    st.sidebar.title("Configuration")
    st.sidebar.info("Apna Firebase project config yahaan paste karein. (Sirf local test ke liye)")
    config_input = st.sidebar.text_area("Firebase Config (JSON format)", 
                                        value=st.session_state.firebase_config_input, 
                                        height=300, 
                                        key="firebase_config_input_widget")
    if config_input:
        firebase_config_json = config_input

# --- Firebase Initialize (Ab ye safe hai) ---
if firebase_config_json and not st.session_state.firebase:
    try:
        config = json.loads(firebase_config_json)
        firebase = pyrebase.initialize_app(config)
        st.session_state.firebase = firebase
        st.session_state.auth = firebase.auth() # Sirf login ke liye use hoga
        st.session_state.firebase_config = config
        
        if 'firebase_config' not in st.secrets: # Local test par message dikhayein
            st.sidebar.success("Firebase Connected! Login/Signup karein.")
        
        st.session_state.firebase_config_input = firebase_config_json
    except Exception as e:
        st.sidebar.error(f"Firebase Error: {e}")
        st.session_state.firebase = None


# Page routing
if st.session_state.page == 'Login' and st.session_state.user:
    st.session_state.page = 'Coach'
if st.session_state.page == 'Coach' and not st.session_state.user:
    st.session_state.page = 'Login'

# --- 1. Login / Signup Page ---
if st.session_state.page == 'Login':
    st.header("Login / Sign Up")
    
    if not st.session_state.firebase:
        st.warning("App Firebase se connect nahi hai.")
    else:
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
                    # Login ke baad, naye REST API se log load karein
                    token = st.session_state.user['idToken']
                    config = st.session_state.firebase_config
                    st.session_state.workout_log = load_workout_logs_rest(config, token)
                    
                    st.session_state.set_counter = 1 # Naye login par set 1 se shuru
                    st.success("Login successful!")
                    safe_speak("Login successful!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Login Error: {e}")

# --- 2. Main Coach Page (Login ke baad) ---
elif st.session_state.page == 'Coach':
    
    # --- Sidebar (Coach Settings) ---
    st.sidebar.title(f"Welcome, {st.session_state.user['email']}")
    
    exercise_choice = st.sidebar.selectbox(
        "Exercise chunein:",
        ("Bicep Curls", "Squats", "Push-ups", "Overhead Press", "Lunges", "Jumping Jacks", "High Knees"),
        key="exercise_choice"
    )

    # Target Reps/Sets feature
    st.sidebar.divider()
    st.sidebar.title("🎯 Set Your Target")
    st.session_state.target_reps = st.sidebar.number_input("Target Reps per Set", min_value=1, value=st.session_state.target_reps)
    st.session_state.target_sets = st.sidebar.number_input("Target Number of Sets", min_value=1, value=st.session_state.target_sets)
    st.sidebar.divider()

    # Smart label logic
    if exercise_choice in ["Bicep Curls", "Push-ups", "Overhead Press"]:
        side_label = "Kaunsa haath track karein?"
    elif exercise_choice in ["Squats", "Lunges", "High Knees"]:
        side_label = "Kaunsa pair track karein?"
    else: # Jumping Jacks
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
    
    # Voice Assistant Settings
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
                f"**Set {log['set_number']} ({log['exercise']})**\n"
                f"Target: {log['target_reps']} | Left: {log['reps_left']} | Right: {log['reps_right']} \n"
                f"Duration: {log['duration']:.0f}s"
            )
            
    # --- Logout Button (Sidebar) ---
    if st.sidebar.button("Logout", use_container_width=True, type="secondary"):
        st.session_state.user = None
        st.session_state.auth = None
        st.session_state.workout_log = []
        st.session_state.set_counter = 1 # Logout par set counter reset
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
            final_reps_left = st.session_state.rep_counter_left
            final_reps_right = st.session_state.rep_counter_right
            final_duration = time.time() - st.session_state.start_time
            
            if final_reps_left > 0 or final_reps_right > 0:
                log_data = {
                    "exercise": exercise_choice,
                    "side": side_choice,
                    "reps_left": final_reps_left,
                    "reps_right": final_reps_right,
                    "duration": final_duration,
                    "set_number": st.session_state.set_counter, # Set number ko log karein
                    "target_reps": st.session_state.target_reps # Target ko log karein
                }
                
                # NAYA LOGIC: Log ko REST API se save karein
                token = st.session_state.user['idToken']
                config = st.session_state.firebase_config
                save_success = save_workout_log_rest(config, token, log_data)
                
                if save_success:
                    # Log ko local list mein bhi add karein (taaki UI turant update ho)
                    st.session_state.workout_log.insert(0, log_data)
                    log_text = f"Set {st.session_state.set_counter} complete! Left: {final_reps_left}, Right: {final_reps_right} reps."
                    st.success(log_text)
                    safe_speak(log_text)
                    
                    # NAYA: Set logic
                    if st.session_state.set_counter < st.session_state.target_sets:
                        st.session_state.set_counter += 1
                        st.info(f"Get ready for Set {st.session_state.set_counter}!")
                        safe_speak(f"Get ready for Set {st.session_state.set_counter}!")
                    else:
                        st.balloons()
                        st.success("Workout Complete! Excellent job!")
                        safe_speak("Workout Complete! Excellent job!")
                        st.session_state.set_counter = 1 # Workout poora, reset
                        
                else:
                    st.error("Workout log save nahi hua (Database Error).")
                    safe_speak("Failed to log workout.")
            else:
                st.warning("Koi rep detect nahi hua. Workout log nahi hua.")
                safe_speak("No reps detected. Workout not logged.")
            
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
                results = pose.process(image_rgb) # Model yahaan run ho raha hai
                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                angle_l, angle_r = 0, 0
                feedback_msg = st.session_state.feedback 
                
                try:
                    if results.pose_landmarks:
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
                            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmask.LEFT_KNEE.value].y]
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
                            angle_l = calculate_angle(hip_l, shoulder_l, wrist_l) # Arm angle
                            
                            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                            angle_r = calculate_angle(hip_r, shoulder_r, wrist_r) # Arm angle

                        # (Thresholds mein koi change nahi)
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
                        current_reps = max(st.session_state.rep_counter_left, st.session_state.rep_counter_right)
                        
                        if current_reps >= st.session_state.target_reps:
                            feedback_msg = "Set Complete! Stop the webcam."
                            # Target poora, reps count karna band karein
                        
                        elif side_choice == 'Left':
                            if angle_l < down_threshold and st.session_state.stage_left == stage_check:
                                st.session_state.stage_left = 'down' if stage_check == 'up' else 'up'
                                st.session_state.rep_counter_left += 1
                                feedback_msg = f'Rep {st.session_state.rep_counter_left}!'
                            elif angle_l > up_threshold and st.session_state.stage_left != stage_check:
                                st.session_state.stage_left = stage_check
                                feedback_msg = 'Ready'

                        elif side_choice == 'Right':
                            if angle_r < down_threshold and st.session_state.stage_right == stage_check:
                                st.session_state.stage_right = 'down' if stage_check == 'up' else 'up'
                                st.session_state.rep_counter_right += 1
                                feedback_msg = f'Rep {st.session_state.rep_counter_right}!'
                            elif angle_r > up_threshold and st.session_state.stage_right != stage_check:
                                st.session_state.stage_right = stage_check
                                feedback_msg = 'Ready'
                        
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
                            elif (stage_l_reset and stage_r_reset and st.session_state.stage != stage_check):
                                st.session_state.stage = stage_check
                                feedback_msg = 'Ready for next rep'
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

                        # Draw landmarks
                        mp_drawing.draw_landmarks(
                            image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                        )
                except Exception as e:
                    st.session_state.feedback = "Poora shareer camera mein dikhayein!"

                # Voice Assistant Logic
                if (st.session_state.voice_enabled and 
                    st.session_state.feedback != st.session_state.last_spoken_feedback):
                    st.session_state.last_spoken_feedback = st.session_state.feedback
                    safe_speak(st.session_state.feedback)

                # Frontend UI: Stats Dikhayein
                stats_placeholder.markdown(f"""
                    <div style="background-color: #222; padding: 15px; border-radius: 10px; font-size: 1.5rem; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                        <div style="text-align: center;">
                            <strong>LEFT REPS</strong><br><span style="color: #00FF00; font-size: 2.5rem;">{st.session_state.rep_counter_left}</span>
                        </div>
                        <div style="text-align: center;">
                            <strong>TIMER</strong><br><span style="color: #00FF00; font-size: 2.5rem;">{int(elapsed_time)}s</span>
                        </div>
                        <div style="text-align: center;">
                            <strong>RIGHT REPS</strong><br><span style="color: #00FF00; font-size: 2.5rem;">{st.session_state.rep_counter_right}</span>
                        </div>
                    </div>
                    
                    <div style="background-color: #222; padding: 15px; border-radius: 10px; font-size: 1.5rem; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                        <div style="text-align: center;">
                            <strong>TARGET REPS</strong><br><span style="color: #00FFFF; font-size: 2.5rem;">{st.session_state.target_reps}</span>
                        </div>
                        <div style="text-align: center;">
                            <strong>CURRENT SET</strong><br><span style="color: #00FFFF; font-size: 2.5rem;">{st.session_state.set_counter} / {st.session_state.target_sets}</span>
                        </div>
                    </div>
                    
                    <div style="font-size: 1.5rem; text-align: center; margin-top: 15px; color: #00FFFF;">
                        <strong>FEEDBACK:</strong> {st.session_state.feedback}
                    </div>
                """, unsafe_allow_html=True)
                
                video_placeholder.image(image_bgr, channels="BGR", width='stretch')
                
                # NAYA FIX 4: Garbage Collection
                # Memory saaf karein taaki app crash na ho
                gc.collect()
                
                if not st.session_state.webcam_started:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            stats_placeholder.empty()
            video_placeholder.empty()
            gc.collect() # Ek baar aur saaf karein
