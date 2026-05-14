"""Reusable Streamlit UI fragments for AI Fitness Coach."""

from pathlib import Path

import streamlit as st


_STYLE_PATH = Path(__file__).resolve().parents[1] / "assets" / "styles.css"


def load_global_styles() -> None:
    """Load the shared CSS stylesheet for the Streamlit app."""
    css = _STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_hero() -> None:
    """Render the top-level landing hero."""
    st.markdown(
        """
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
        unsafe_allow_html=True,
    )


def render_login_value_card() -> None:
    """Render the left-side product value card on the auth screen."""
    st.markdown(
        """
        <div class="smart-card">
            <h2 style="margin-top:0;">Why this coach feels smarter</h2>
            <p>A polished full-stack fitness app with real-time computer vision, Supabase-backed accounts, workout history, nutrition planning, and AI coaching.</p>
            <div class="pill-row">
                <span class="pill">✅ 7 exercises</span>
                <span class="pill">✅ Rep + set tracking</span>
                <span class="pill">✅ Secure profiles</span>
                <span class="pill">✅ CSV export</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auth_header(is_configured: bool) -> None:
    """Render the auth panel heading and provider status."""
    auth_provider = "Supabase" if is_configured else "Not configured"
    status_class = "status-ok" if is_configured else "status-bad"
    st.markdown(
        f"""
        <div class="auth-panel">
            <div class="auth-title">Login / Sign Up</div>
            <p class="auth-muted">Auth Provider: <span class="{status_class}">{auth_provider}</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
