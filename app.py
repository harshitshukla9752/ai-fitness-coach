import cv2
import mediapipe as mp
import streamlit as st
import time
from utils import calculate_angle
import streamlit.components.v1 as components
import json
import gc  # Memory optimization
import os
import io
import csv
from datetime import datetime, timezone
from collections import defaultdict
import pandas as pd
from openai import OpenAI
from supabase import create_client

st.set_page_config(
    page_title="AI Fitness Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Voice Assistant (TTS) Function ---
def speak(text, lang, voice_name):
    # Remove quotes that can break the injected JavaScript string.
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
            // Wait for browser voices to load.
            if (window.speechSynthesis.getVoices().length > 0) {{ doSpeak(); }}
            else {{ window.speechSynthesis.onvoiceschanged = doSpeak; }}
        </script>
    """
    components.html(speech_js, height=0, width=0)

# --- Cache AI Models (Memory Optimization) ---
@st.cache_resource
def load_models():
    import os
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Prefer bundled model path when available, otherwise fall back to user cache path.
    project_dir = os.path.dirname(os.path.abspath(__file__))
    bundled_model = os.path.join(project_dir, "models", "pose_landmark_lite.tflite")
    cache_model_dir = os.path.expanduser("~/.mediapipe/modules/pose_landmark/")
    cache_model = os.path.join(cache_model_dir, "pose_landmark_lite.tflite")

    local_model = bundled_model if os.path.exists(bundled_model) else cache_model
    os.environ["MEDIAPIPE_MODEL_PATH"] = os.path.dirname(local_model)

    if not os.path.exists(local_model):
        st.error(
            "Pose model not found. Expected at ./models/pose_landmark_lite.tflite "
            "or ~/.mediapipe/modules/pose_landmark/pose_landmark_lite.tflite."
        )
        st.stop()

    try:
        # Try normal initialization first
        pose = mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        st.info("✅ MediaPipe pose model loaded successfully from local cache.")
        return mp_pose, pose, mp_drawing

    except Exception as first_err:
        # If initialization failed (often due to download blocked), try to detect
        # a locally cached model file and give clear guidance to the user.
        st.warning(f"MediaPipe init warning: {first_err}")

        # Common cache locations
        cache_paths = [
            os.path.join(os.path.expanduser('~'), '.cache', 'mediapipe'),
            os.path.join(os.path.expanduser('~'), '.mediapipe'),
            '/tmp/mediapipe',
            './models'
        ]

        found_local = False
        for p in cache_paths:
            if os.path.exists(p):
                # if directory exists, we assume the model might be present
                found_local = True
                break

        if not found_local:
            # Provide actionable error and stop app: downloading at runtime likely blocked.
            st.error(
                "MediaPipe model initialization failed and no local model cache was found. "
                "If you're deploying to a managed host (Streamlit Cloud, Heroku, etc.), "
                "pre-download the MediaPipe model during the build step and place it in "
                "one of these paths: ~/.cache/mediapipe, ~/.mediapipe, /tmp/mediapipe or ./models. "
                "See README for exact filename (pose_landmark_*.tflite)."
            )
            st.stop()

        # If a cache path exists, try a second time (MediaPipe may find the file now)
        try:
            pose = mp_pose.Pose(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            return mp_pose, pose, mp_drawing
        except Exception as second_err:
            st.error(
                f"MediaPipe still failed to initialize after checking cache paths:\n{second_err}\n" 
                "Make sure the model .tflite has been downloaded to the cache or allow network access."
            )
            st.stop()

# Load cached models.
mp_pose, pose, mp_drawing = load_models()

# --- Session State Initialization ---
def _init_default_states():
    default_states = {
        'voice_enabled': True, 'voice_lang': 'hi-IN', 'voice_name': 'Google हिन्दी',
        'workout_log': [], 'rep_counter_left': 0, 'rep_counter_right': 0, 'set_counter': 1,
        'stage_left': 'down', 'stage_right': 'down', 'stage': 'down',
        'feedback': 'Start your workout!', 'last_spoken_feedback': '',
        'start_time': 0, 'webcam_started': False,
        'firebase_config_input': '', 'firebase_config': None, 'firebase': None,
        'auth': None, 'user': None, 'page': 'Login',
        'target_reps': 10, 'target_sets': 3, 'workout_complete_feedback_given': False,
        'goal_focus': 'Hypertrophy (Muscle Gain)',
        'ai_chat_history': [],
        'challenge_mode': False,
        'ai_last_reply': '',
        'ai_auto_voice': True,
        'use_real_ai': True,
        'profile_age': 22,
        'profile_gender': 'Male',
        'profile_height_cm': 170,
        'profile_weight_kg': 70,
        'profile_body_type': 'Average',
        'profile_activity_level': 'Moderately Active',
        'profile_goal_type': 'Body Recomposition (Fat Loss + Muscle Gain)',
        'profile_diet_type': 'Vegetarian',
        'profile_experience_level': 'Beginner',
        'profile_loaded': False,
        'supabase': None,
        'use_supabase_auth': False
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

_init_default_states()

# --- Helper functions (Speak, Reset, TTS) ---
def safe_speak(text):
    if st.session_state.voice_enabled:
        speak(text, st.session_state.voice_lang, st.session_state.voice_name)

def reset_states(exercise_choice):
    st.session_state.rep_counter_left = 0
    st.session_state.rep_counter_right = 0
    # Keep the set counter running across the current workout.
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
    st.session_state.workout_complete_feedback_given = False  # Reset feedback for the new set.

VOICE_OPTIONS = {
    'hi-IN': {
        'Hindi Voice (Default)': 'Google हिन्दी',
        'Hindi Voice (Female)': 'Microsoft Kalpana - Hindi (India)'
    },
    'en-US': {
        'English (Female - Default)': 'Google US English',
        'English (Female - Realistic)': 'Microsoft Zira - English (United States)',
        'English (Male - Realistic)': 'Microsoft David - English (United States)'
    }
}

def init_supabase():
    secrets_url = st.secrets.get("SUPABASE_URL")
    secrets_key = st.secrets.get("SUPABASE_ANON_KEY")

    # Support nested Streamlit secrets format:
    # [supabase]
    # url = "..."
    # key = "..."
    supabase_block = st.secrets.get("supabase", {})
    block_url = None
    block_key = None
    if hasattr(supabase_block, "get"):
        block_url = supabase_block.get("url")
        block_key = supabase_block.get("key") or supabase_block.get("anon_key")

    # User requirement: load Supabase credentials from Streamlit secrets only.
    url = secrets_url or block_url
    key = secrets_key or block_key
    if url:
        url = str(url).strip().rstrip("/")
    if key:
        key = str(key).strip()
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        st.warning(f"Supabase init failed: {e}")
        return None

def format_supabase_auth_error(action, error):
    error_text = str(error)
    lowered = error_text.lower()
    if "getaddrinfo" in lowered or "name resolution" in lowered or "temporary failure" in lowered:
        return (
            f"{action} Error: The Supabase URL cannot be reached through DNS/network. "
            "Set the correct `[supabase] url` project URL in `.streamlit/secrets.toml`, "
            "check internet/DNS access, and restart the app."
        )
    if "invalid login credentials" in lowered:
        return f"{action} Error: The email or password is incorrect."
    if "email not confirmed" in lowered:
        return f"{action} Error: Please verify the Supabase confirmation email first."
    return f"{action} Error: {error_text}"

def save_workout_log_supabase(log_data):
    try:
        payload = {**log_data, "user_id": st.session_state.user["localId"]}
        st.session_state.supabase.table("workout_logs").insert(payload).execute()
        return True
    except Exception as e:
        st.error(f"Supabase save error: {e}")
        return False

def load_workout_logs_supabase():
    try:
        uid = st.session_state.user["localId"]
        result = (
            st.session_state.supabase.table("workout_logs")
            .select("*")
            .eq("user_id", uid)
            .order("timestamp", desc=True)
            .execute()
        )
        rows = result.data if result and result.data else []
        logs = []
        for row in rows:
            logs.append({
                "exercise": row.get("exercise", "N/A"),
                "side": row.get("side", "N/A"),
                "reps_left": int(row.get("reps_left", 0)),
                "reps_right": int(row.get("reps_right", 0)),
                "duration": float(row.get("duration", 0.0)),
                "set_number": int(row.get("set_number", 1)),
                "target_reps": int(row.get("target_reps", 10)),
                "timestamp": row.get("timestamp", "")
            })
        return logs
    except Exception as e:
        st.error(f"Supabase load error: {e}")
        return []

def save_user_profile_supabase(profile_data):
    try:
        payload = {**profile_data, "user_id": st.session_state.user["localId"]}
        st.session_state.supabase.table("user_profiles").upsert(payload, on_conflict="user_id").execute()
        return True
    except Exception as e:
        st.warning(f"Supabase profile save issue: {e}")
        return False

def load_user_profile_supabase():
    try:
        uid = st.session_state.user["localId"]
        result = st.session_state.supabase.table("user_profiles").select("*").eq("user_id", uid).limit(1).execute()
        rows = result.data if result and result.data else []
        if not rows:
            return None
        row = rows[0]
        return {
            "age": row.get("age"),
            "gender": row.get("gender"),
            "height_cm": row.get("height_cm"),
            "weight_kg": row.get("weight_kg"),
            "body_type": row.get("body_type"),
            "activity_level": row.get("activity_level"),
            "goal_type": row.get("goal_type"),
            "diet_type": row.get("diet_type"),
            "experience_level": row.get("experience_level")
        }
    except Exception as e:
        st.warning(f"Supabase profile load issue: {e}")
        return None

# -----------------------------------------------------------------
# --- END OF NEW DATABASE LOGIC ---
# -----------------------------------------------------------------

def parse_timestamp(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def safe_index(options, value, default=0):
    try:
        return options.index(value)
    except ValueError:
        return default

def calculate_analytics(logs):
    if not logs:
        return None
    parsed_logs = []
    for log in logs:
        ts = parse_timestamp(log.get("timestamp", ""))
        total_reps = log.get("reps_left", 0) + log.get("reps_right", 0)
        parsed_logs.append({
            **log,
            "timestamp_dt": ts,
            "total_reps": total_reps,
            "intensity": round(total_reps / max(log.get("duration", 1), 1), 2)
        })

    total_sessions = len(parsed_logs)
    total_reps_all = sum(item["total_reps"] for item in parsed_logs)
    total_duration = sum(item.get("duration", 0) for item in parsed_logs)
    avg_intensity = round(sum(item["intensity"] for item in parsed_logs) / total_sessions, 2)

    exercise_summary = defaultdict(lambda: {"sessions": 0, "reps": 0, "duration": 0})
    daily_reps = defaultdict(int)
    for item in parsed_logs:
        ex = item.get("exercise", "Unknown")
        exercise_summary[ex]["sessions"] += 1
        exercise_summary[ex]["reps"] += item["total_reps"]
        exercise_summary[ex]["duration"] += item.get("duration", 0)
        if item["timestamp_dt"]:
            day_key = item["timestamp_dt"].date().isoformat()
            daily_reps[day_key] += item["total_reps"]

    active_days = sorted(daily_reps.keys())
    streak = 0
    if active_days:
        day_objs = [datetime.fromisoformat(d).date() for d in active_days]
        streak = 1
        for i in range(len(day_objs) - 1, 0, -1):
            if (day_objs[i] - day_objs[i-1]).days == 1:
                streak += 1
            else:
                break

    summary_df = pd.DataFrame([
        {
            "Exercise": ex,
            "Sessions": data["sessions"],
            "Total Reps": data["reps"],
            "Total Duration (s)": round(data["duration"], 1),
            "Avg Reps / Session": round(data["reps"] / max(data["sessions"], 1), 1)
        }
        for ex, data in exercise_summary.items()
    ]).sort_values(by="Total Reps", ascending=False)

    trend_df = pd.DataFrame([
        {"Date": day, "Total Reps": reps}
        for day, reps in sorted(daily_reps.items())
    ])

    return {
        "total_sessions": total_sessions,
        "total_reps_all": total_reps_all,
        "total_duration": total_duration,
        "avg_intensity": avg_intensity,
        "streak": streak,
        "summary_df": summary_df,
        "trend_df": trend_df
    }

def suggest_targets(goal_focus, current_reps, current_sets, analytics):
    reps, sets_ = current_reps, current_sets
    if goal_focus == "Strength":
        reps = max(4, min(10, current_reps - 2))
        sets_ = min(6, current_sets + 1)
    elif goal_focus == "Endurance":
        reps = min(30, current_reps + 4)
        sets_ = min(6, current_sets + 1)
    else:  # Hypertrophy
        reps = min(15, max(8, current_reps + 1))
        sets_ = min(5, max(3, current_sets))

    if analytics and analytics["avg_intensity"] < 0.35:
        reps = max(6, reps - 1)
    elif analytics and analytics["avg_intensity"] > 1.0:
        reps = min(30, reps + 1)

    return reps, sets_

def logs_to_csv(logs):
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["timestamp", "exercise", "side", "reps_left", "reps_right", "duration", "set_number", "target_reps"]
    )
    writer.writeheader()
    for log in logs:
        writer.writerow(log)
    return output.getvalue()

def generate_ai_insights(analytics, logs, goal_focus):
    if not analytics:
        return [
            "AI Insight: Your baseline is still building. Complete 3-5 sessions consistently first.",
            "AI Insight: Focus on form and tempo first, then increase load or reps.",
        ]

    insights = []
    if analytics["streak"] >= 5:
        insights.append("AI Insight: Excellent consistency! You can now apply progressive overload safely.")
    else:
        insights.append("AI Insight: Improve consistency. Set a minimum 3-day streak challenge.")

    if analytics["avg_intensity"] < 0.35:
        insights.append("AI Insight: Intensity is low. Add 1-2 reps per set with controlled tempo.")
    elif analytics["avg_intensity"] > 1.1:
        insights.append("AI Insight: Intensity is high. Prioritize recovery, hydration, and sleep.")
    else:
        insights.append("AI Insight: Intensity is balanced. Maintain this pattern for the next week.")

    if logs:
        top = calculate_analytics(logs)["summary_df"].iloc[0]["Exercise"]
        insights.append(f"AI Insight: Your best-performing movement is **{top}**. Keep it as an anchor exercise.")

    if goal_focus == "Strength":
        insights.append("Goal Strategy: Prioritize low-rep, high-quality sets on compound movements.")
    elif goal_focus == "Endurance":
        insights.append("Goal Strategy: Maintain short rest intervals (30-45 sec) and higher reps.")
    else:
        insights.append("Goal Strategy: Use the 8-15 rep zone with strict form and strong mind-muscle connection.")

    return insights

def generate_weekly_plan(goal_focus, challenge_mode=False):
    base_plan = {
        "Day 1": "Upper Body Focus + Core",
        "Day 2": "Lower Body Strength + Mobility",
        "Day 3": "Cardio + Active Recovery",
        "Day 4": "Push-Pull Mixed Session",
        "Day 5": "Leg Endurance + Stability",
        "Day 6": "HIIT / Conditioning",
        "Day 7": "Full Rest + Stretching"
    }

    if goal_focus == "Strength":
        modifier = " (Low reps, longer rest, strict execution)"
    elif goal_focus == "Endurance":
        modifier = " (High reps, short rest, continuous flow)"
    else:
        modifier = " (Moderate reps, controlled eccentric tempo)"

    if challenge_mode:
        modifier += " + Daily finisher challenge"

    return {day: f"{task}{modifier}" for day, task in base_plan.items()}

def ai_coach_reply(user_query, analytics, goal_focus):
    q = user_query.lower().strip()
    if not q:
        return "Type your question about form, reps, diet, or recovery."
    if "diet" in q or "protein" in q or "khana" in q:
        return "Prioritize lean protein, complex carbs, hydration, and a post-workout meal within 30-60 minutes."
    if "recovery" in q or "sleep" in q:
        return "Recovery rule: sleep 7-8 hours, do light mobility, and give the same muscle group 48 hours to recover."
    if "plateau" in q or "ruk" in q:
        return "Plateau break: increase only one variable at a time, such as load or reps, and plan a deload week."
    if "goal" in q or "plan" in q:
        return f"For your current goal, **{goal_focus}**, follow target progression and the weekly split planner."
    if analytics and analytics["avg_intensity"] < 0.35:
        return "Based on your data, your intensity needs a boost. Try +1 rep per set next session."
    return "Use a form-first approach: full range of motion, controlled tempo, and consistent weekly progression."

def get_real_ai_response(user_query, analytics, goal_focus, calorie_data=None, diet_type=None, experience_level=None):
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY missing"
    try:
        client = OpenAI(api_key=api_key)
        context = {
            "goal_focus": goal_focus,
            "analytics": analytics if analytics else {},
            "calorie_data": calorie_data if calorie_data else {},
            "diet_type": diet_type,
            "experience_level": experience_level
        }
        system_prompt = (
            "You are FitGPT, a dedicated AI coach for this specific project: an AI Virtual Fitness Coach app. "
            "Always provide practical, personalized workout + nutrition advice based on given context. "
            "Use clear English with concise bullets. "
            "Cover reps/sets progression, recovery, calories/macros, and diet options (veg/vegan/non-veg) if relevant. "
            "If user asks unsafe/extreme recommendations, guide toward safe moderate approach."
        )
        user_prompt = (
            f"User question: {user_query}\n\n"
            f"Project context JSON: {json.dumps(context, default=str)}\n\n"
            "Give actionable answer for this user only, not generic."
        )
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return completion.choices[0].message.content, None
    except Exception as e:
        return None, str(e)

def calculate_calories_and_macros(age, gender, height_cm, weight_kg, activity_level, goal_type):
    if gender == "Male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    activity_map = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Athlete": 1.9
    }
    tdee = bmr * activity_map.get(activity_level, 1.55)

    if goal_type == "Fat Loss":
        target_cal = tdee - 450
        protein = weight_kg * 2.0
        fats = weight_kg * 0.8
    elif goal_type == "Muscle Gain":
        target_cal = tdee + 320
        protein = weight_kg * 1.8
        fats = weight_kg * 1.0
    else:  # Recomposition
        target_cal = tdee
        protein = weight_kg * 2.0
        fats = weight_kg * 0.9

    protein_cals = protein * 4
    fat_cals = fats * 9
    carbs = max(0, (target_cal - protein_cals - fat_cals) / 4)

    return {
        "bmr": round(bmr),
        "tdee": round(tdee),
        "target_calories": round(target_cal),
        "protein_g": round(protein),
        "carbs_g": round(carbs),
        "fats_g": round(fats)
    }

def get_diet_recommendations(diet_type, goal_type, experience_level):
    plans = {
        "Vegetarian": {
            "breakfast": ["Oats + Milk + Banana + Peanut Butter", "Cottage cheese scramble + multigrain flatbread"],
            "lunch": ["Lentils + rice + salad + yogurt", "Kidney beans + cumin rice + vegetable bowl"],
            "snacks": ["Roasted chickpeas + fruit", "Greek yogurt + nuts"],
            "dinner": ["Tofu or cottage cheese stir-fry + flatbread", "Soy protein curry + quinoa"]
        },
        "Vegan": {
            "breakfast": ["Soy milk oats + chia seeds", "Chickpea flour pancakes + peanut dip"],
            "lunch": ["Chickpea salad + millet roti", "Lentil bowl + brown rice + veggies"],
            "snacks": ["Seasoned sprouts bowl", "Peanut butter toast + fruit"],
            "dinner": ["Tofu curry + quinoa", "Soya granules + mixed vegetables"]
        },
        "Non-Vegetarian": {
            "breakfast": ["Egg omelette + toast + fruit", "Greek yogurt + oats + nuts"],
            "lunch": ["Chicken breast + rice + salad", "Fish + sweet potato + veggies"],
            "snacks": ["Boiled eggs + black coffee", "Whey shake + banana"],
            "dinner": ["Chicken soup + flatbread + salad", "Egg curry + brown rice"]
        }
    }
    level_tip = {
        "Beginner": "Beginner Tip: 80% consistency beats perfection. Repeating simple meals is completely fine.",
        "Intermediate": "Intermediate Tip: Optimize meal timing around your workout.",
        "Pro / Advanced Athlete": "Pro Tip: Track weekly refeeds, sodium-potassium balance, and peri-workout nutrition."
    }
    goal_tip = {
        "Fat Loss": "Goal Tip: 400-500 calorie deficit + high protein + daily steps 8k+.",
        "Muscle Gain": "Goal Tip: Use a lean bulk with a controlled calorie surplus of 200-350 kcal.",
        "Body Recomposition (Fat Loss + Muscle Gain)": "Goal Tip: Prioritize strength progression, high protein, and consistent sleep."
    }
    return plans[diet_type], level_tip[experience_level], goal_tip[goal_type]

def get_training_blueprint(experience_level):
    if experience_level == "Beginner":
        return {
            "split": "3-4 days Full Body / Upper-Lower",
            "volume": "8-12 sets per muscle/week",
            "intensity": "RPE 6-7, form first",
            "progression": "Add 1-2 reps each week or increase load by 2.5%"
        }
    if experience_level == "Intermediate":
        return {
            "split": "4-5 days Push-Pull-Legs / Upper-Lower",
            "volume": "12-16 sets per muscle/week",
            "intensity": "RPE 7-8",
            "progression": "Double progression model + deload every 5-7 weeks"
        }
    return {
        "split": "5-6 days periodized split",
        "volume": "14-22 sets per muscle/week (phase dependent)",
        "intensity": "RPE 8-9 with planned recovery",
        "progression": "Block periodization + fatigue management + performance tracking"
    }


# --- Main App ---
st.markdown(
    """
    <style>
        :root {
            --bg: #070b14;
            --panel: rgba(15, 23, 42, 0.78);
            --panel-strong: rgba(15, 23, 42, 0.94);
            --border: rgba(148, 163, 184, 0.22);
            --text: #f8fafc;
            --muted: #cbd5e1;
            --accent: #38bdf8;
            --accent-2: #22c55e;
            --danger: #fb7185;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.22), transparent 34rem),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.15), transparent 28rem),
                linear-gradient(135deg, #070b14 0%, #111827 48%, #0f172a 100%);
            color: var(--text);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, h4, h5, h6, p, label, span {color: var(--text);}
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] * {color: var(--text);}
        .hero-card, .auth-card, .smart-card, .metric-card {
            border: 1px solid var(--border);
            border-radius: 24px;
            background: var(--panel);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(18px);
        }
        .hero-card {
            padding: 34px;
            margin-bottom: 22px;
            position: relative;
            overflow: hidden;
        }
        .hero-card:before {
            content: "";
            position: absolute;
            inset: -80px -120px auto auto;
            width: 280px;
            height: 280px;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.18);
            filter: blur(4px);
        }
        .hero-title {
            font-size: clamp(2.3rem, 6vw, 4.8rem);
            line-height: 0.95;
            font-weight: 900;
            letter-spacing: -0.06em;
            margin: 0 0 18px 0;
        }
        .hero-subtitle {
            color: var(--muted);
            font-size: 1.08rem;
            max-width: 780px;
            margin-bottom: 22px;
        }
        .pill-row {display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px;}
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.06);
            color: var(--text);
            font-weight: 700;
            font-size: 0.88rem;
        }
        .auth-card {padding: 24px; margin-top: 4px;}
        .auth-title {font-size: 1.45rem; font-weight: 850; margin-bottom: 6px;}
        .auth-muted {color: var(--muted); margin-bottom: 18px;}
        .status-ok {color: #86efac; font-weight: 800;}
        .status-bad {color: #fca5a5; font-weight: 800;}
        .smart-card {padding: 16px; margin-bottom: 12px;}
        .metric-card {padding: 18px; text-align: center;}
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(56, 189, 248, 0.45);
            background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
            color: white;
            font-weight: 800;
            min-height: 44px;
        }
        div.stTextInput > div > div > input, div.stNumberInput input,
        div[data-baseweb="select"] > div, textarea {
            border-radius: 14px !important;
            border: 1px solid rgba(148, 163, 184, 0.45) !important;
            background: #ffffff !important;
            color: #0f172a !important;
            min-height: 44px;
            box-shadow: 0 10px 28px rgba(2, 6, 23, 0.18);
        }
        [data-baseweb="input"], [data-baseweb="input"] *,
        div[data-baseweb="select"] > div, div[data-baseweb="select"] > div *,
        div[data-baseweb="popover"] *, div[role="listbox"] * {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
        }
        div.stTextInput input::placeholder,
        div.stNumberInput input::placeholder {
            color: #475569 !important;
            -webkit-text-fill-color: #475569 !important;
            opacity: 1;
        }
        input, textarea,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            caret-color: #0f172a !important;
        }
        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInput"] button * {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
        }
        div[data-baseweb="select"] svg,
        [data-baseweb="input"] svg {fill: #0f172a !important; color: #0f172a !important;}
        div[data-baseweb="popover"] > div,
        div[role="listbox"] {
            background: #ffffff !important;
            border: 1px solid rgba(15, 23, 42, 0.16) !important;
            border-radius: 14px !important;
        }
        [data-testid="stAlert"] {
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.28);
            background: rgba(15, 23, 42, 0.9);
            color: #f8fafc;
        }
        [data-testid="stTabs"] button {color: #e2e8f0; font-weight: 800;}
        [data-testid="stTabs"] button[aria-selected="true"] {color: #38bdf8;}
        .stDataFrame, [data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        .live-stats-card {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.28);
            padding: 18px;
            border-radius: 22px;
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.22);
        }
        .live-stat {text-align: center; padding: 12px; border-radius: 18px; background: rgba(255,255,255,0.06);}
        .live-stat strong {display:block; color:#cbd5e1; font-size:0.85rem; letter-spacing:0.06em;}
        .live-stat span {display:block; color:#67e8f9; font-size:clamp(2rem, 5vw, 3rem); font-weight:900;}
        .feedback-card {
            margin-top: 12px;
            padding: 16px;
            border-radius: 18px;
            text-align: center;
            color: #f8fafc;
            background: linear-gradient(135deg, rgba(14,165,233,0.24), rgba(34,197,94,0.18));
            border: 1px solid rgba(103, 232, 249, 0.35);
            font-size: 1.25rem;
            font-weight: 800;
        }
        @media (max-width: 760px) {
            .live-stats-card {grid-template-columns: 1fr;}
            .hero-card {padding: 24px;}
        }
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 14px;
        }
    </style>
    <div class="hero-card">
        <div class="pill">🏋️ AI Fitness Coach • Supabase Powered</div>
        <h1 class="hero-title">Train smarter.<br/>Track every rep.</h1>
        <p class="hero-subtitle">Real-time pose detection, guided reps, progress analytics, AI planning, and secure cloud history in one polished coaching dashboard.</p>
        <div class="pill-row">
            <span class="pill">📹 Live Pose Detection</span>
            <span class="pill">📊 Analytics Dashboard</span>
            <span class="pill">🧠 AI Training Planner</span>
            <span class="pill">🔐 Supabase Auth + DB</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Deployment Logic (Supabase only) ---
# Credentials are read silently from .streamlit/secrets.toml.
# No login-screen configuration sidebar is shown.
if st.session_state.supabase is None:
    st.session_state.supabase = init_supabase()
st.session_state.use_supabase_auth = st.session_state.supabase is not None

# Page routing
if st.session_state.page == 'Login' and st.session_state.user:
    st.session_state.page = 'Coach'
if st.session_state.page == 'Coach' and not st.session_state.user:
    st.session_state.page = 'Login'

# --- 1. Login / Signup Page ---
if st.session_state.page == 'Login':
    left_col, auth_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="smart-card">
                <h2 style="margin-top:0;">Why this coach feels smarter</h2>
                <p style="color:#cbd5e1;">A polished full-stack fitness app with real-time computer vision, Supabase-backed accounts, workout history, nutrition planning, and AI coaching.</p>
                <div class="pill-row">
                    <span class="pill">✅ 7 exercises</span>
                    <span class="pill">✅ Rep + set tracking</span>
                    <span class="pill">✅ Secure profiles</span>
                    <span class="pill">✅ CSV export</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Exercises", "7")
        c2.metric("Storage", "Supabase")
        c3.metric("Coach", "AI + Voice")

    with auth_col:
        st.markdown('<div class="auth-title">Login / Sign Up</div>', unsafe_allow_html=True)
        auth_provider = "Supabase" if st.session_state.use_supabase_auth else "Not configured"
        status_class = "status-ok" if st.session_state.use_supabase_auth else "status-bad"
        st.markdown(
            f'<p class="auth-muted">Auth Provider: <span class="{status_class}">{auth_provider}</span></p>',
            unsafe_allow_html=True
        )

        if not st.session_state.use_supabase_auth:
            st.error("Supabase is not configured. Add credentials to `.streamlit/secrets.toml`, then restart the app.")
        else:
            choice = st.radio("Choose action", ("Login", "Sign Up"), horizontal=True, label_visibility="collapsed")
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")

            if choice == "Sign Up":
                if st.button("Create account", use_container_width=True):
                    if not email or not password:
                        st.warning("Email and password are required.")
                    else:
                        try:
                            resp = st.session_state.supabase.auth.sign_up({"email": email, "password": password})
                            user_obj = resp.user
                            session_obj = resp.session
                            st.session_state.user = {
                                "email": user_obj.email if user_obj else email,
                                "localId": user_obj.id if user_obj else email,
                                "idToken": session_obj.access_token if session_obj else ""
                            }
                            st.rerun()
                        except Exception as e:
                            st.error(format_supabase_auth_error("Signup", e))

            if choice == "Login":
                if st.button("Login", use_container_width=True):
                    if not email or not password:
                        st.warning("Email and password are required.")
                    else:
                        try:
                            resp = st.session_state.supabase.auth.sign_in_with_password({"email": email, "password": password})
                            user_obj = resp.user
                            session_obj = resp.session
                            st.session_state.user = {
                                "email": user_obj.email,
                                "localId": user_obj.id,
                                "idToken": session_obj.access_token if session_obj else ""
                            }
                            st.session_state.workout_log = load_workout_logs_supabase()
                            profile = load_user_profile_supabase()

                            if profile:
                                st.session_state.profile_age = profile.get("age", st.session_state.profile_age)
                                st.session_state.profile_gender = profile.get("gender", st.session_state.profile_gender)
                                st.session_state.profile_height_cm = profile.get("height_cm", st.session_state.profile_height_cm)
                                st.session_state.profile_weight_kg = profile.get("weight_kg", st.session_state.profile_weight_kg)
                                st.session_state.profile_body_type = profile.get("body_type", st.session_state.profile_body_type)
                                st.session_state.profile_activity_level = profile.get("activity_level", st.session_state.profile_activity_level)
                                st.session_state.profile_goal_type = profile.get("goal_type", st.session_state.profile_goal_type)
                                st.session_state.profile_diet_type = profile.get("diet_type", st.session_state.profile_diet_type)
                                st.session_state.profile_experience_level = profile.get("experience_level", st.session_state.profile_experience_level)
                            st.session_state.profile_loaded = True

                            st.session_state.set_counter = 1
                            st.rerun()
                        except Exception as e:
                            st.error(format_supabase_auth_error("Login", e))

# --- 2. Main Coach Page (after login) ---
elif st.session_state.page == 'Coach':
    
    # --- Sidebar (Coach Settings) ---
    st.sidebar.title(f"Welcome, {st.session_state.user['email'].split('@')[0]}")
    
    exercise_choice = st.sidebar.selectbox(
        "Choose Exercise:",
        ("Bicep Curls", "Squats", "Push-ups", "Overhead Press", "Lunges", "Jumping Jacks", "High Knees"),
        key="exercise_choice"
    )

    # Target Reps/Sets feature
    st.sidebar.divider()

    st.sidebar.title("🧠 Smart Goal Focus")
    st.session_state.goal_focus = st.sidebar.selectbox(
        "Training Goal",
        ("Hypertrophy (Muscle Gain)", "Strength", "Endurance"),
        index=("Hypertrophy (Muscle Gain)", "Strength", "Endurance").index(st.session_state.goal_focus)
    )
    st.sidebar.divider()
    st.session_state.challenge_mode = st.sidebar.checkbox("Enable Challenge Mode", value=st.session_state.challenge_mode)
    st.session_state.use_real_ai = st.sidebar.checkbox("Use Real AI Coach (OpenAI)", value=st.session_state.use_real_ai)
    st.sidebar.divider()
    st.sidebar.title("🎯 Set Your Target")
    st.session_state.target_reps = st.sidebar.number_input("Target Reps per Set", min_value=1, value=st.session_state.target_reps)
    st.session_state.target_sets = st.sidebar.number_input("Target Number of Sets", min_value=1, value=st.session_state.target_sets)
    st.sidebar.divider()

    # Smart label logic
    if exercise_choice in ["Bicep Curls", "Push-ups", "Overhead Press"]:
        side_label = "Which arm should be tracked?"
    elif exercise_choice in ["Squats", "Lunges", "High Knees"]:
        side_label = "Which leg should be tracked?"
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
    lang_map = {'Hindi Voice': 'hi-IN', 'English Voice': 'en-US'}
    default_lang_index = 0 if st.session_state.voice_lang == 'hi-IN' else 1
    selected_lang_friendly = st.sidebar.selectbox("Assistant Voice Language:", ('Hindi Voice', 'English Voice'), index=default_lang_index)
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
        st.sidebar.info("No workout logs yet.")
    else:
        for i, log in enumerate(st.session_state.workout_log):
            ts = parse_timestamp(log.get("timestamp", ""))
            ts_text = ts.strftime("%d %b %Y, %I:%M %p UTC") if ts else "No Timestamp"
            st.sidebar.success(
                f"**Set {log['set_number']} ({log['exercise']})**\n"
                f"Target: {log['target_reps']} | Left: {log['reps_left']} | Right: {log['reps_right']} \n"
                f"Duration: {log['duration']:.0f}s \n"
                f"When: {ts_text}"
            )
            
    # --- Logout Button (Sidebar) ---
    if st.sidebar.button("Logout", use_container_width=True, type="secondary"):
        if st.session_state.supabase:
            try:
                st.session_state.supabase.auth.sign_out()
            except Exception:
                pass
        st.session_state.user = None
        st.session_state.auth = None
        st.session_state.workout_log = []
        st.session_state.set_counter = 1 # Reset the set counter on logout.
        st.success("Logged out successfully!")
        safe_speak("Logged out successfully!")
        time.sleep(1)
        st.rerun()

    # --- Main App Interface (Coach) ---
    st.caption(f"Selected workout: **{exercise_choice} ({side_choice})**.")
    analytics = calculate_analytics(st.session_state.workout_log)
    tab_live, tab_analytics, tab_plan = st.tabs(["🎥 Live Coach", "📈 Advanced Analytics", "🧠 AI Training Planner"])

    with tab_live:
        if st.button("Start / Stop Webcam", key="start_stop_button", use_container_width=True, type="primary"):
            st.session_state.webcam_started = not st.session_state.webcam_started
            
            if st.session_state.webcam_started:
                # Reset the set counter if needed.
                if st.session_state.set_counter > st.session_state.target_sets:
                    st.session_state.set_counter = 1
                reset_states(exercise_choice)
            
            elif not st.session_state.webcam_started and st.session_state.start_time != 0:
                # Stop logic
                final_reps_left = st.session_state.rep_counter_left
                final_reps_right = st.session_state.rep_counter_right
                final_duration = time.time() - st.session_state.start_time
                timestamp_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                
                if final_reps_left > 0 or final_reps_right > 0:
                    log_data = {
                        "exercise": exercise_choice,
                        "side": side_choice,
                        "reps_left": final_reps_left,
                        "reps_right": final_reps_right,
                        "duration": final_duration,
                        "set_number": st.session_state.set_counter, # Log the set number.
                        "target_reps": st.session_state.target_reps, # Log the target.
                        "timestamp": timestamp_now
                    }
                    
                    save_success = save_workout_log_supabase(log_data)
                    
                    if save_success:
                        # Add the log locally so the UI updates immediately.
                        st.session_state.workout_log.insert(0, log_data)
                        log_text = f"Set {st.session_state.set_counter} complete! Left: {final_reps_left}, Right: {final_reps_right} reps."
                        st.success(log_text)
                        safe_speak(log_text)
                        
                        # Set progression logic.
                        if st.session_state.set_counter < st.session_state.target_sets:
                            st.session_state.set_counter += 1
                            st.info(f"Get ready for Set {st.session_state.set_counter}!")
                            safe_speak(f"Get ready for Set {st.session_state.set_counter}!")
                        else:
                            st.balloons()
                            st.success("Workout Complete! Excellent job!")
                            safe_speak("Workout Complete! Excellent job!")
                            st.session_state.set_counter = 1 # Workout complete, reset.
                            
                    else:
                        st.error("Workout log was not saved because of a database error.")
                        safe_speak("Failed to log workout.")
                else:
                    st.warning("No reps were detected, so the workout was not logged.")
                    safe_speak("No reps detected. Workout not logged.")
                
                st.session_state.start_time = 0
                st.rerun()

    with tab_analytics:
        st.subheader("Performance Intelligence Dashboard")
        if not analytics:
            st.info("Workout data is needed for analytics. Complete 2-3 sets to unlock insights.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Sessions", analytics["total_sessions"])
            c2.metric("Total Reps", analytics["total_reps_all"])
            c3.metric("Workout Time", f"{int(analytics['total_duration'] // 60)} min")
            c4.metric("Current Streak", f"{analytics['streak']} day(s)")
            st.caption(f"Estimated Intensity Score: **{analytics['avg_intensity']} reps/sec**")
            st.dataframe(analytics["summary_df"], use_container_width=True, hide_index=True)
            if not analytics["trend_df"].empty:
                st.line_chart(analytics["trend_df"].set_index("Date")["Total Reps"], use_container_width=True)
            csv_data = logs_to_csv(st.session_state.workout_log)
            st.download_button(
                "⬇️ Download Workout Data (CSV)",
                data=csv_data,
                file_name="workout_logs.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown("### 🤖 AI Performance Insights")
            for msg in generate_ai_insights(analytics, st.session_state.workout_log, st.session_state.goal_focus):
                st.markdown(f"<div class='smart-card'>{msg}</div>", unsafe_allow_html=True)

    with tab_plan:
        st.subheader("Adaptive Training Plan")
        recommended_reps, recommended_sets = suggest_targets(
            st.session_state.goal_focus,
            st.session_state.target_reps,
            st.session_state.target_sets,
            analytics
        )
        st.info(
            f"Goal: **{st.session_state.goal_focus}**\n\n"
            f"AI Recommendation → Next target: **{recommended_sets} sets × {recommended_reps} reps**"
        )
        if st.button("Apply Recommended Target", use_container_width=True):
            st.session_state.target_reps = recommended_reps
            st.session_state.target_sets = recommended_sets
            st.success("Recommended targets applied to sidebar settings.")

        st.markdown("### Form & Progress Suggestions")
        if analytics:
            top_exercise = analytics["summary_df"].iloc[0]["Exercise"]
            st.success(f"Strongest pattern: **{top_exercise}** — keep it as a weekly priority.")
            if analytics["streak"] < 3:
                st.warning("Consistency is low. Start a 3-day streak challenge to build momentum.")
            else:
                st.success("Great consistency! Continue progressive overload safely.")
        st.markdown(
            """
            - Warmup 5-7 min before session.
            - Maintain a full range of motion on every rep.
            - Follow the 48-hour recovery rule for the same muscle group.
            - Improve one measurable metric each week (reps, form, or control).
            """
        )

        st.markdown("### 🗓️ Smart Weekly Program Generator")
        weekly_plan = generate_weekly_plan(st.session_state.goal_focus, st.session_state.challenge_mode)
        for day, plan in weekly_plan.items():
            st.markdown(f"<div class='smart-card'><strong>{day}</strong><br>{plan}</div>", unsafe_allow_html=True)

        st.markdown("### 🧬 Body Type + Goal Based Smart Nutrition Engine")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            age = st.number_input("Age", min_value=14, max_value=75, value=st.session_state.profile_age)
            gender_options = ["Male", "Female"]
            gender = st.selectbox("Gender", gender_options, index=safe_index(gender_options, st.session_state.profile_gender))
            level_options = ["Beginner", "Intermediate", "Pro / Advanced Athlete"]
            experience_level = st.selectbox("Level", level_options, index=safe_index(level_options, st.session_state.profile_experience_level))
        with col_b:
            height_cm = st.number_input("Height (cm)", min_value=130, max_value=220, value=st.session_state.profile_height_cm)
            weight_kg = st.number_input("Weight (kg)", min_value=35, max_value=180, value=st.session_state.profile_weight_kg)
            body_type_options = ["High Body Fat", "Lean / Skinny", "Average"]
            body_type = st.selectbox("Body Type", body_type_options, index=safe_index(body_type_options, st.session_state.profile_body_type))
        with col_c:
            activity_options = ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Athlete"]
            activity_level = st.selectbox("Activity", activity_options, index=safe_index(activity_options, st.session_state.profile_activity_level))
            goal_options = ["Fat Loss", "Muscle Gain", "Body Recomposition (Fat Loss + Muscle Gain)"]
            goal_type = st.selectbox(
                "Transformation Goal",
                goal_options,
                index=safe_index(goal_options, st.session_state.profile_goal_type)
            )
            diet_options = ["Vegetarian", "Vegan", "Non-Vegetarian"]
            diet_type = st.selectbox("Diet Preference", diet_options, index=safe_index(diet_options, st.session_state.profile_diet_type))

        st.session_state.profile_age = age
        st.session_state.profile_gender = gender
        st.session_state.profile_height_cm = height_cm
        st.session_state.profile_weight_kg = weight_kg
        st.session_state.profile_body_type = body_type
        st.session_state.profile_activity_level = activity_level
        st.session_state.profile_goal_type = goal_type
        st.session_state.profile_diet_type = diet_type
        st.session_state.profile_experience_level = experience_level

        if st.button("💾 Save Profile for Next Login", use_container_width=True):
            profile_payload = {
                "age": age,
                "gender": gender,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "body_type": body_type,
                "activity_level": activity_level,
                "goal_type": goal_type,
                "diet_type": diet_type,
                "experience_level": experience_level,
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            saved = save_user_profile_supabase(profile_payload)
            if saved:
                st.success("Profile saved. It will auto-load on your next login.")

        calorie_data = calculate_calories_and_macros(age, gender, height_cm, weight_kg, activity_level, goal_type)
        meal_plan, level_tip, goal_tip = get_diet_recommendations(diet_type, goal_type, experience_level)
        blueprint = get_training_blueprint(experience_level)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("BMR", f"{calorie_data['bmr']} kcal")
        m2.metric("TDEE", f"{calorie_data['tdee']} kcal")
        m3.metric("Target Calories/Day", f"{calorie_data['target_calories']} kcal")
        m4.metric("Body Type", body_type)
        st.caption(
            f"Daily Macros → Protein: **{calorie_data['protein_g']}g**, Carbs: **{calorie_data['carbs_g']}g**, Fats: **{calorie_data['fats_g']}g**"
        )

        st.markdown(
            f"""
            <div class='smart-card'>
            <strong>Training Blueprint ({experience_level})</strong><br>
            Split: {blueprint['split']}<br>
            Weekly Volume: {blueprint['volume']}<br>
            Intensity: {blueprint['intensity']}<br>
            Progression: {blueprint['progression']}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='smart-card'><strong>{goal_tip}</strong><br>{level_tip}</div>",
            unsafe_allow_html=True
        )

        meal_col1, meal_col2 = st.columns(2)
        with meal_col1:
            st.markdown("#### Meal Suggestions")
            st.write(f"**Breakfast:** {', '.join(meal_plan['breakfast'])}")
            st.write(f"**Lunch:** {', '.join(meal_plan['lunch'])}")
        with meal_col2:
            st.markdown("#### Snacks & Dinner")
            st.write(f"**Snacks:** {', '.join(meal_plan['snacks'])}")
            st.write(f"**Dinner:** {', '.join(meal_plan['dinner'])}")

        if st.button("🔊 Speak Nutrition & Plan Summary", use_container_width=True):
            summary_text = (
                f"Your target calories are {calorie_data['target_calories']} per day. "
                f"Protein {calorie_data['protein_g']} grams, carbs {calorie_data['carbs_g']} grams, fats {calorie_data['fats_g']} grams. "
                f"Goal is {goal_type}. Diet preference is {diet_type}."
            )
            safe_speak(summary_text)

        st.markdown("### 💬 Ask AI Coach")
        if st.session_state.use_real_ai:
            st.caption("Real AI mode is on. Set `OPENAI_API_KEY` in Streamlit secrets or as an environment variable.")
        st.session_state.ai_auto_voice = st.checkbox("Auto-speak AI replies", value=st.session_state.ai_auto_voice)
        user_query = st.text_input("Ask about form, progression, recovery, diet...", key="ai_query")
        if st.button("Get AI Coach Reply", use_container_width=True):
            response = None
            if st.session_state.use_real_ai:
                response, err = get_real_ai_response(
                    user_query=user_query,
                    analytics=analytics,
                    goal_focus=st.session_state.goal_focus,
                    calorie_data=calorie_data,
                    diet_type=diet_type,
                    experience_level=experience_level
                )
                if err:
                    st.warning(f"Real AI unavailable ({err}). Fallback coach answer shown.")
            if not response:
                response = ai_coach_reply(user_query, analytics, st.session_state.goal_focus)
            st.session_state.ai_last_reply = response
            st.session_state.ai_chat_history.insert(0, {"q": user_query, "a": response})
            if st.session_state.ai_auto_voice:
                safe_speak(response)

        if st.button("🔊 Speak Last AI Reply", use_container_width=True) and st.session_state.ai_last_reply:
            safe_speak(st.session_state.ai_last_reply)

        if st.session_state.ai_chat_history:
            for chat in st.session_state.ai_chat_history[:5]:
                st.markdown(
                    f"<div class='smart-card'><strong>You:</strong> {chat['q']}<br><strong>Coach:</strong> {chat['a']}</div>",
                    unsafe_allow_html=True
                )

    # --- Stats and Video Placeholders (Frontend UI) ---
    stats_placeholder = st.empty()
    video_placeholder = st.empty()

    # --- Main Backend Loop ---
    if st.session_state.webcam_started:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam could not start. Check camera permissions.")
        else:
            while st.session_state.webcam_started:
                ret, frame = cap.read()
                if not ret:
                    break

                elapsed_time = time.time() - st.session_state.start_time
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)  # Run the pose model on the frame.
                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                angle_l, angle_r = 0, 0
                feedback_msg = st.session_state.feedback 
                
                try:
                    # Check whether the workout is complete.
                    if st.session_state.set_counter > st.session_state.target_sets:
                        if not st.session_state.workout_complete_feedback_given:
                            st.session_state.feedback = "Workout Complete! Stop the webcam."
                            safe_speak(st.session_state.feedback)
                            st.session_state.workout_complete_feedback_given = True
                        
                        # Draw landmarks while skipping rep logic.
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            )
                    
                    elif results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        
                        # Exercise-specific pose logic.
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
                            angle_l = calculate_angle(hip_l, shoulder_l, wrist_l) # Arm angle
                            
                            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                            angle_r = calculate_angle(hip_r, shoulder_r, wrist_r) # Arm angle

                        # Exercise-specific motion thresholds.
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
                        current_reps_left = st.session_state.rep_counter_left
                        current_reps_right = st.session_state.rep_counter_right
                        
                        # Reps check
                        reps_met_left = current_reps_left >= st.session_state.target_reps
                        reps_met_right = current_reps_right >= st.session_state.target_reps
                        reps_met_both = reps_met_left and reps_met_right

                        if (side_choice == 'Left' and reps_met_left) or \
                           (side_choice == 'Right' and reps_met_right) or \
                           (side_choice == 'Both' and reps_met_both):
                            
                            feedback_msg = f"Set {st.session_state.set_counter} Complete! Stop webcam to save."
                            if not st.session_state.workout_complete_feedback_given: # Sirf ek baar bolo
                                safe_speak(feedback_msg)
                                st.session_state.workout_complete_feedback_given = True
                        
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
                                if not reps_met_both:
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
                except Exception:
                    st.session_state.feedback = "Keep your full body visible in the camera frame!"

                # Voice Assistant Logic
                if (st.session_state.voice_enabled and 
                    st.session_state.feedback != st.session_state.last_spoken_feedback):
                    st.session_state.last_spoken_feedback = st.session_state.feedback
                    safe_speak(st.session_state.feedback)

                # Frontend UI: show stats.
                stats_placeholder.markdown(f"""
                    <div class="live-stats-card">
                        <div class="live-stat">
                            <strong>LEFT REPS</strong><span>{st.session_state.rep_counter_left}</span>
                        </div>
                        <div class="live-stat">
                            <strong>TIMER</strong><span>{int(elapsed_time)}s</span>
                        </div>
                        <div class="live-stat">
                            <strong>RIGHT REPS</strong><span>{st.session_state.rep_counter_right}</span>
                        </div>
                        <div class="live-stat">
                            <strong>TARGET REPS</strong><span>{st.session_state.target_reps}</span>
                        </div>
                        <div class="live-stat">
                            <strong>CURRENT SET</strong><span>{st.session_state.set_counter} / {st.session_state.target_sets}</span>
                        </div>
                        <div class="live-stat">
                            <strong>STATUS</strong><span>Live</span>
                        </div>
                    </div>
                    <div class="feedback-card">FEEDBACK: {st.session_state.feedback}</div>
                """, unsafe_allow_html=True)
                
                video_placeholder.image(image_bgr, channels="BGR", width='stretch')
                
                # Memory cleanup.
                # Free memory to reduce crash risk.
                gc.collect()
                
                if not st.session_state.webcam_started:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            stats_placeholder.empty()
            video_placeholder.empty()
            gc.collect()  # Final memory cleanup.
