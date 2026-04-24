# Supabase Setup (Auth + Database)

## 1) Create project and enable Email auth
1. Open Supabase dashboard and create a new project.
2. Go to **Authentication → Providers → Email** and keep Email/Password enabled.

## 2) Create DB schema
1. Open **SQL Editor**.
2. Run the file: `supabase/schema.sql`.
3. Confirm tables are created:
   - `public.workout_logs`
   - `public.user_profiles`

## 3) Add app secrets
Set these values in Streamlit secrets (or environment variables):

```toml
SUPABASE_URL = "https://<project-ref>.supabase.co"
SUPABASE_ANON_KEY = "<anon-public-key>"
```

## 4) Verify in app
1. Start app: `streamlit run app.py`
2. Create a new account using Login page.
3. Complete one set and click save.
4. Check Supabase table `workout_logs` for inserted row.
5. Save profile once and verify `user_profiles` upsert.

## Notes
- The app writes `user_id` as the authenticated user UUID from Supabase auth.
- Row Level Security policies in `schema.sql` ensure users can only access their own rows.
