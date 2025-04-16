import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import streamlit as st
from typing import List, Any

@st.cache_resource
def get_sheets_service():
    """Cache Google Sheets service configuration"""
    ##  try:
    ##    creds = service_account.Credentials.from_service_account_file( # type: ignore
    ##        os.getenv('GOOGLE_SHEETS_CREDENTIALS'), 
    ##        scopes=['https://www.googleapis.com/auth/spreadsheets']
    ##   )
    ##    service: Any = build('sheets', 'v4', credentials=creds)
    ##    return service
    ##except Exception as e:
    ##    raise e

    try:
        # Try secrets-based credentials (for Streamlit Cloud)
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        print("Credentials loaded")  # Just for sanity check
        return build("sheets", "v4", credentials=credentials)
    except Exception as e:
        print(f"❌ Failed to load credentials from secrets: {e}")
        return None  # Prevents crashing in Home.py
