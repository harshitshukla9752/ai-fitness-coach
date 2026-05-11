# Project Structure

```text
ai-fitness-coach/
├── app.py                         # Main Streamlit app: UI, auth flow, live coach, analytics, AI planner
├── utils.py                       # Shared pose/angle math helpers
├── requirements.txt               # Python runtime dependencies
├── packages.txt                   # System packages for hosted deployments
├── setup.sh                       # Deployment setup script for MediaPipe model cache
├── models/                        # Bundled MediaPipe model assets
│   └── pose_landmark_lite.tflite
├── supabase/                      # Supabase project files
│   ├── config.toml                # Supabase CLI project configuration
│   ├── migrations/                # Versioned database migrations
│   │   └── 20260510000000_initial_schema.sql
│   ├── schema.sql                 # Manual SQL Editor fallback schema
│   └── README.md                  # Supabase setup and verification guide
├── .streamlit/
│   └── secrets.example.toml       # Local secrets template; copy to secrets.toml
└── docs/
    └── PROJECT_STRUCTURE.md       # Architecture and folder guide
```

## Runtime flow

1. Streamlit loads Supabase credentials from `.streamlit/secrets.toml`.
2. Users sign up or log in through Supabase Auth.
3. Workout logs are inserted into `public.workout_logs`.
4. Profile data is upserted into `public.user_profiles`.
5. Row Level Security keeps each user's data private.

## Database workflow

Use Supabase CLI for repeatable setup:

```bash
supabase link --project-ref <project-ref>
supabase db push
```

If CLI is not available, run `supabase/schema.sql` manually in Supabase SQL Editor.
