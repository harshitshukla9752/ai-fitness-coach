# Supabase Setup (Auth + Database)

## 1) Create project and enable Email auth
1. Open Supabase dashboard and create a new project.
2. Go to **Authentication → Providers → Email** and keep Email/Password enabled.

## 2) Apply the database migration

### Recommended: Supabase CLI
This repo includes `supabase/config.toml` and a real migration file. Run these commands from the project root after logging in to Supabase CLI:

```bash
supabase link --project-ref <project-ref>
supabase db push
```

The migration file is:

```text
supabase/migrations/20260510000000_initial_schema.sql
```

### Manual fallback: SQL Editor
If you are not using Supabase CLI, open **SQL Editor** and run `supabase/schema.sql` manually.

After applying the migration, confirm these tables exist:
- `public.workout_logs`
- `public.user_profiles`

## 3) Add app secrets
Set these values in `.streamlit/secrets.toml`:

```toml
[supabase]
url = "https://<project-ref>.supabase.co"
key = "<anon-public-key>"
```

Restart Streamlit after changing the secrets file.

## 4) Verify in app
1. Start app: `streamlit run app.py`
2. Create a new account using Login page.
3. Complete one set and click save.
4. Check Supabase table `workout_logs` for inserted row.
5. Save profile once and verify `user_profiles` upsert.

## Notes
- The app writes `user_id` as the authenticated user UUID from Supabase auth.
- Row Level Security policies in the migration ensure users can only access their own rows.
