import streamlit as st
import bcrypt
import yaml
from yaml.loader import SafeLoader
from pathlib import Path

# Load credentials from config
with open(Path(__file__).parent.parent / "config" / "config.yaml", "r") as f:
    config = yaml.load(f, Loader=SafeLoader)

# Add your custom CSS here
st.markdown("""
<style>
/* === Background & Font === */
html, body, .stApp {
    background-color: #fffbea !important;
    font-family: 'Segoe UI', sans-serif;
}

/* === Sidebar Gradient Match === */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom right, #fdf2d5, #fae1aa) !important;
    border-right: 2px solid #e5c97b !important;
    box-shadow: 2px 0px 8px rgba(0, 0, 0, 0.05);
    padding: 1rem;
}

/* === Sidebar Button (Logout / Login) === */
.stButton > button {
    background: linear-gradient(135deg, #f1c40f, #f39c12);
    color: white !important;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 10px 18px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e67e22, #f39c12);
    transform: scale(1.02);
}

/* === Login Form Area === */
[data-testid="stForm"] {
    background-color: #fffdf2 !important;
    border: 1px solid #ffe69a !important;
    border-radius: 16px;
    padding: 2rem;
    max-width: 480px;
    margin: 8vh auto;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

/* === Inputs === */
input, .stTextInput input, .stPasswordInput input {
    background-color: #fffef8 !important;
    border: 1px solid #f4d276 !important;
    padding: 12px;
    border-radius: 8px;
    font-size: 16px;
}

/* === Welcome Message === */
h1, h2, h3 {
    color: #b37a00;
    text-align: center;
}

/* === Notification Box Styling === */
.stAlert {
    border-radius: 8px;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("### 🔐 Login to Smart Tracker")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        user = config['credentials']['usernames'].get(username)
        if user and bcrypt.checkpw(password.encode(), user['password'].encode()):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success(f"✅ Welcome {user['name']} 👋")
        else:
            st.session_state["authenticated"] = False
            st.error("❌ Incorrect username or password. Please try again.")
    st.stop()

# --- SHOW APP IF AUTHENTICATED --- #
else:
    name = config['credentials']['usernames'][st.session_state["username"]]['name']
    st.sidebar.success(f"✅ Welcome, {name}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()