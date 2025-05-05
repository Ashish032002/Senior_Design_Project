import streamlit as st
import bcrypt
import yaml
from yaml.loader import SafeLoader
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

# Load config
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.load(f, Loader=SafeLoader)

# Save config
def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

config = load_config()

st.markdown("""
<style>
html, body, .stApp {
    background-color: #fffbea !important;
    font-family: 'Segoe UI', sans-serif;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom right, #fdf2d5, #fae1aa) !important;
    border-right: 2px solid #e5c97b !important;
}
.stButton > button {
    background: linear-gradient(135deg, #f1c40f, #f39c12);
    color: white !important;
    font-weight: bold;
}
[data-testid="stForm"] {
    background-color: #fffdf2 !important;
    border: 1px solid #ffe69a !important;
    border-radius: 16px;
    padding: 2rem;
    max-width: 480px;
    margin: 8vh auto;
}
</style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

# --- LOGIN TAB --- #
with tab1:
    st.markdown("### 🔐 Login to Smart Tracker")
    with st.form("login_form"):
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
            st.error("❌ Incorrect username or password.")

# --- REGISTER TAB --- #
with tab2:
    st.markdown("### 📝 Register New User")
    with st.form("register_form"):
        new_username = st.text_input("New Username")
        full_name = st.text_input("Full Name")
        email = st.text_input("Email")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        register_btn = st.form_submit_button("Register")

    if register_btn:
        if new_username in config["credentials"]["usernames"]:
            st.warning("⚠️ Username already exists.")
        elif new_password != confirm_password:
            st.warning("⚠️ Passwords do not match.")
        else:
            hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
            config["credentials"]["usernames"][new_username] = {
                "name": full_name,
                "email": email,
                "password": hashed_pw
            }
            save_config(config)
            st.success("✅ User registered! You can now log in.")

# --- SHOW APP IF AUTHENTICATED --- #
if st.session_state["authenticated"]:
    name = config['credentials']['usernames'][st.session_state["username"]]['name']
    st.sidebar.success(f"✅ Welcome, {name}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()
