import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from pathlib import Path



st.set_page_config(
    page_title="Smart Finance Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.warning("🔒 Please login to continue.")
    st.stop()

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()


from logging import Logger
from typing import Any
import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import sys
from dateutil import parser
import re
from config.constants import TRANSACTION_TYPES, CATEGORIES
from services.google_sheets import get_sheets_service                
from utils.logging_utils import setup_logging
from pathlib import Path
from services.google_sheets import get_sheets_service
import yfinance as yf


log: Logger = setup_logging("expense_tracker")

# Load environment variables
load_dotenv()
log.info("Environment variables loaded")


st.markdown("""
    <style>
    
    /* === Base and Background === */
    html, body, [class*="css"] {
        background-color: #fffbea !important;
        color: #2c2c2c;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #fffbea;
    }

    /* === Sidebar Styling === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom right, #fdf2d5, #fae1aa);
        border-right: 2px solid #e5c97b;
        box-shadow: 2px 0px 8px rgba(0, 0, 0, 0.05);
    }
    .stSidebar > div {
        padding: 20px 10px;
    }
    .sidebar .element-container {
        padding: 10px;
    }

    /* === Title and Instructions === */
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #b37a00;
        padding-bottom: 8px;
    }
    .instruction {
        font-size: 1.1rem;
        color: #946b00;
        padding-bottom: 15px;
    }

    /* === Chat Bubbles === */
    .stChatMessage.user {
        background: #fff6d9;
        border-left: 4px solid #f9be37;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(249, 190, 55, 0.2);
    }
    .stChatMessage.assistant {
        background: #fef3cc;
        border-left: 4px solid #f1c40f;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(241, 196, 15, 0.2);
    }

    /* === Buttons === */
    .stButton > button {
        background: linear-gradient(135deg, #f1c40f, #f39c12);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 15px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #e67e22, #f39c12);
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15);
    }

    /* === Form Container === */
    .stForm {
        background-color: #fffdf2;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        border: 1px solid #ffe69a;
    }

    /* === Select & Input === */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div {
        background-color: #fffef8;
        border: 1px solid #f4d276;
        padding: 10px;
        border-radius: 6px;
    }

    /* === Metrics === */
    .stMetric {
        background-color: #fffbde;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border: 1px solid #ffe69a;
        text-align: center;
    }

    /* === Dataframe === */
    .stDataFrame {
        border: 1px solid #ffe69a;
        background-color: #fffef4;
        border-radius: 10px;
        overflow: hidden;
    }

    /* === Plotly Charts === */
    .js-plotly-plot {
       background-color: transparent !important;
       border-radius: 0px;

    }

    /* === Scrollbar Customization === */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #f1c40f;
        border-radius: 10px;
    }

    .block-container {
        padding-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)




# --- Sidebar ---
st.sidebar.title("💼 Personal Finance Tracker")
st.sidebar.markdown("Welcome! 👋\nLog your income and expenses in a smart, conversational way.")
if "show_analytics" not in st.session_state:
    st.session_state.show_analytics = False
    
with st.sidebar.expander("📈 Trending Stocks"):
    st.markdown("- TCS")
    st.markdown("- INFY")
    st.markdown("- RELIANCE")
    st.markdown("- AAPL")
    st.markdown("- MSFT")
    st.markdown("- GOOGL")
    st.markdown("_Try asking: 'What's the price of TCS today?'_")

if st.sidebar.button("📊 Toggle Analytics View"):
    st.session_state.show_analytics = not st.session_state.show_analytics
log.info(f"Analytics view toggled: {st.session_state.show_analytics}")
    
st.markdown("""
    <style>

    /* === Base and Background === */
    html, body, [class*="css"] {
        background-color: #fffbea !important;
        color: #2c2c2c;
        font-family: 'Segoe UI', sans-serif;
    }

    .stApp,
    .main,
    .block-container,
    .css-1dp5vir,
    .css-ffhzg2 {
        background-color: #fffbea !important;
    }

    /* === Header and Footer Background Fix === */
    header[data-testid="stHeader"],
    footer {
        background-color: #fffbea !important;
        box-shadow: none;
    }

    /* === Sidebar Styling === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom right, #fdf2d5, #fae1aa);
        border-right: 2px solid #e5c97b;
        box-shadow: 2px 0px 8px rgba(0, 0, 0, 0.05);
    }

    .stSidebar > div {
        padding: 20px 10px;
    }

    .sidebar .element-container {
        padding: 10px;
    }

    /* === Title and Instructions === */
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #b37a00;
        padding-bottom: 8px;
    }

    .instruction {
        font-size: 1.1rem;
        color: #946b00;
        padding-bottom: 15px;
    }

    /* === Chat Bubbles === */
    .stChatMessage.user {
        background: #fff6d9;
        border-left: 4px solid #f9be37;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(249, 190, 55, 0.2);
    }

    .stChatMessage.assistant {
        background: #fef3cc;
        border-left: 4px solid #f1c40f;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(241, 196, 15, 0.2);
    }

    /* === Buttons === */
    .stButton > button {
        background: linear-gradient(135deg, #f1c40f, #f39c12);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 15px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #e67e22, #f39c12);
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15);
    }

    /* === Form Container === */
    .stForm {
        background-color: #fffdf2;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        border: 1px solid #ffe69a;
    }

    /* === Select & Input === */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div {
        background-color: #fffef8;
        border: 1px solid #f4d276;
        padding: 10px;
        border-radius: 6px;
    }

    /* === Metrics === */
    .stMetric {
        background-color: #fffbde;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border: 1px solid #ffe69a;
        text-align: center;
    }

    /* === Dataframe === */
    .stDataFrame {
        border: 1px solid #ffe69a;
        background-color: #fffef4;
        border-radius: 10px;
        overflow: hidden;
    }

    /* === Plotly Charts === */
    .js-plotly-plot {
       background-color: transparent !important;
       border-radius: 0px;
    }

    /* === Scrollbar Customization === */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-thumb {
        background: #f1c40f;
        border-radius: 10px;
    }

    .block-container {
        padding-top: 1.5rem;
    }

    /* === Chat Input Area Fix === */
    .stChatInputContainer {
        background-color: #fffbea !important;
        border-top: 2px solid #f4d276 !important;
        padding: 1rem !important;
    }

    input[data-baseweb="textarea"],
    textarea[data-baseweb="textarea"] {
        background-color: #fffef2 !important;
        color: #2c2c2c !important;
        border: 2px solid #f5d26c !important;
        border-radius: 20px !important;
        padding: 10px 16px !important;
        font-size: 16px !important;
        box-shadow: none !important;
    }

    input[data-baseweb="textarea"]:focus,
    textarea[data-baseweb="textarea"]:focus {
        border-color: #f1c40f !important;
        outline: none !important;
    }

    .stChatInputContainer button {
        background-color: #f1c40f !important;
        color: white !important;
        border-radius: 50% !important;
        border: none !important;
        padding: 8px !important;
        transition: 0.3s ease;
    }

    .stChatInputContainer button:hover {
        background-color: #e67e22 !important;
    }

    </style>
""", unsafe_allow_html=True)



st.markdown('<div class="main-title">💬 Smart Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="instruction">Type a message like "I spent 500 on groceries yesterday" or just say hi!</div>', unsafe_allow_html=True)
st.divider()


@st.cache_resource
def get_gemini_model() -> Any:
    """Cache Gemini AI configuration"""
    try:
       
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")

        genai.configure(api_key=api_key)
        model: Any = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore
        log.info("Gemini AI configured successfully")
        return model
    except Exception as e:
        log.error(f"❌ Failed to configure Gemini AI: {str(e)}")
        raise



try:
    model = get_gemini_model()
    service = get_sheets_service()

    
    try:
        SHEET_ID = st.secrets["GOOGLE_SHEET_ID"]
    except Exception:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        SHEET_ID = os.getenv("GOOGLE_SHEET_ID")

    log.info("Google Sheets API connected successfully")
except Exception as e:
    log.error(f"Failed to connect to Google Sheets: {str(e)}")
    log.error(f"Failed to initialize services: {str(e)}")
    sys.exit(1)



@st.cache_data(ttl=300)
def get_categories() -> dict[str, dict[str, list[str]]]:
    """Cache the categories dictionary to prevent reloading"""
    return CATEGORIES

@st.cache_data
def get_transaction_types() -> list[str]:
    """Cache the transaction types to prevent reloading"""
    return TRANSACTION_TYPES

def init_session_state() -> None:
    """
    Initialize Streamlit session state variables with default values.
    Sets up necessary state variables for the application.
    """
    defaults: dict[str, Any] = {
        'messages': [],
        'save_clicked': False,
        'current_amount': None,
        'current_type': None,
        'current_category': None,
        'current_subcategory': None,
        'form_submitted': False,
        'show_analytics': False,  # New state variable for analytics
        'current_transaction': None,
        'stock_result': None,
        
        # New state variable for current transaction
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def parse_date_from_text(text: str) -> datetime:
    """
    Extract and parse date from input text.
    
    Args:
        text (str): Input text containing date information
        
    Returns:
        str: Parsed date in YYYY-MM-DD format
    """
    current_date: datetime = datetime.now()
    try:
        text = text.lower()
        
        relative_dates: dict[str, datetime] = {
            'today': current_date,
            'yesterday': current_date - timedelta(days=1),
            'tomorrow': current_date + timedelta(days=1),
            'day before yesterday': current_date - timedelta(days=2),
        }

        for phrase, date in relative_dates.items():
            if phrase in text:
                return date

        last_pattern: str = r'last (\d+) (day|week|month)s?'
        match: re.Match[str] | None = re.search(last_pattern, text)
        if match:
            number: int = int(match.group(1))
            unit: str | Any = match.group(2)
            if unit == 'day':
                return current_date - timedelta(days=number)
            elif unit == 'week':
                return current_date - timedelta(weeks=number)
            elif unit == 'month':
                return current_date - timedelta(days=number * 30)

        next_pattern = r'next (\d+) (day|week|month)s?'
        match = re.search(next_pattern, text)
        if match:
            number = int(match.group(1))
            unit = match.group(2)
            if unit == 'day':
                return current_date + timedelta(days=number)
            elif unit == 'week':
                return current_date + timedelta(weeks=number)
            elif unit == 'month':
                return current_date + timedelta(days=number * 30)

        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        match = re.search(date_pattern, text)
        if match:
            return parser.parse(match.group())
        words: list[str] = text.split()
        for i in range(len(words)-2):
            possible_date: str = ' '.join(words[i:i+3])
            try:
                return parser.parse(possible_date)
            except Exception as e:
                log.error(f"Failed to parse date from text: {str(e)}")
                continue
        
        return current_date
    
    except Exception as e:
        log.warning(f"Failed to parse date from text, using current date. Error: {str(e)}")
        return current_date
    



STOCK_NAME_MAP = {
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "reliance": "RELIANCE.NS",
    "hdfc": "HDFCBANK.NS",
    "apple": "AAPL",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "microsoft": "MSFT",
    "meta": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA"
}

def detect_stock_ticker(text: str) -> str | None:
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)

    for word in words:
        if word in STOCK_NAME_MAP:
            return STOCK_NAME_MAP[word]

    # Try detecting uppercase tickers like GOOGL or AAPL
    match = re.search(r'\b([A-Z]{3,6})\b', text.upper())
    if match:
        symbol = match.group(1)
        known_us_tickers = list(STOCK_NAME_MAP.values()) + ["GOOGL", "AAPL", "AMZN", "MSFT", "META", "TSLA", "NVDA"]
        if symbol in known_us_tickers:
            return symbol
        else:
            return symbol + ".NS"

    return None
  


@st.cache_data(ttl=3600)
def get_stock_data(ticker: str) -> dict[str, Any]:
    try:
        stock = yf.Ticker(ticker)

        # Fetch historical data
        hist = stock.history(period="1mo")
        if hist is None or hist.empty:
            log.error(f"❌ No historical data found for: {ticker}")
            return {}

        # Try fetching stock info safely
        try:
            info = stock.info
        except Exception as e:
            log.error(f"❌ Failed to fetch stock.info for {ticker}: {e}")
            return {}

        if not info or "shortName" not in info:
            log.error(f"❌ Invalid stock info for: {ticker}")
            return {}

        # Return cleaned info
        return {
            "name": info.get("shortName", ticker),
            "price": info.get("currentPrice", hist['Close'].iloc[-1]),
            "currency": info.get("currency", "INR"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "eps": info.get("trailingEps", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "history": hist
        }

    except Exception as e:
        log.error(f"🚨 Stock fetch failed: {str(e)}")
        return {}


def generate_stock_advice(prompt: str, model: Any) -> str:
    try:
        chat = model.start_chat(history=[])
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        log.warning(f"Gemini stock advice failed: {str(e)}")
        return "Unable to generate stock advice at this time."


def test_sheet_access() -> bool:
    """
    Test Google Sheets API connection.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        # Test write access by appending to the last row instead of clearing

        test_values: list[list[str]] = [['TEST', 'TEST', 'TEST', 'TEST', 'TEST', 'TEST']]
        result: Any = service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range='Expenses',
            valueInputOption='RAW',
            body={'values': test_values}
        ).execute()
        
        # Get the range that was just written
        updated_range:str = result['updates']['updatedRange']
        
        # Only clear the test row we just added
        service.spreadsheets().values().clear(
            spreadsheetId=SHEET_ID,
            range=updated_range,
            body={}
        ).execute()
        
        log.info("Sheet access test successful")
        return True
    except Exception as e:
        log.error(f"Sheet access test failed: {str(e)}")
        return False

def initialize_sheet() -> None:
    try:
        # Create sheets if they don't exist
        sheet_metadata: Any = service.spreadsheets().get(spreadsheetId=SHEET_ID).execute() # type: ignore
        sheets: list[Any] = sheet_metadata.get('sheets', '') # type: ignore
        existing_sheets: set[Any] = {s.get("properties", {}).get("title") for s in sheets} # type: ignore
        
        # Initialize Expenses sheet
        if 'Expenses' not in existing_sheets:
            log.info("Creating new Expenses sheet...")
            body: dict[str, Any] = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': 'Expenses'
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate( # type: ignore
                spreadsheetId=SHEET_ID,
                body=body
            ).execute()
            
            headers: list[list[str]] = [['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description']]
            service.spreadsheets().values().update( # type: ignore
                spreadsheetId=SHEET_ID,
                range='Expenses!A1:F1',
                valueInputOption='RAW',
                body={'values': headers}
            ).execute()
        
        
        if 'Pending' not in existing_sheets:
            log.info("Creating new Pending sheet...")
            body: dict[str, Any] = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': 'Pending'
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate( # type: ignore
                spreadsheetId=SHEET_ID,
                body=body
            ).execute()
            
            headers: list[list[str]] = [['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status']]
            service.spreadsheets().values().update( # type: ignore
                spreadsheetId=SHEET_ID,
                range='Pending!A1:G1',
                valueInputOption='RAW',
                body={'values': headers}
            ).execute()
        
        # Test sheet access
        if not test_sheet_access():
            raise Exception("Failed to verify sheet access")
            
        log.info("Sheets initialized and verified")
    except Exception as e:
        log.error(f"Failed to initialize sheets: {str(e)}")
        raise

def add_transaction_to_sheet(date: str, amount: float, trans_type: str, 
                           category: str, subcategory: str, description: str) -> bool:
    """
    Add a new transaction to Google Sheet.
    
    Args:
        date (str): Transaction date in YYYY-MM-DD format
        amount (float): Transaction amount
        trans_type (str): Type of transaction (Income/Expense)
        category (str): Transaction category
        subcategory (str): Transaction subcategory
        description (str): Transaction description
        
    Returns:
        bool: True if transaction added successfully, False otherwise
    """
    try:
        log.info(f"Starting transaction save: {date}, {amount}, {trans_type}, {category}, {subcategory}, {description}")
        
        # Format the date if it's a datetime object
        date_str:Any = date
        
        # Ensure amount is a string
        amount_str: str = str(float(amount))
        
        # Prepare the values
        values: list[list[str]] = [[str(date_str), amount_str, trans_type, category, subcategory, description]]
        
        # Changed range to 'Expenses' to let Google Sheets determine the next empty row
        result: Any = service.spreadsheets().values().append( # type: ignore
            spreadsheetId=SHEET_ID,
            range='Expenses',  # Changed from 'Expenses!A2:F2' to just 'Expenses'
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body={'values': values}
        ).execute()
        
        log.info(f"Transaction saved successfully: {result}")
        return True
        
    except Exception as e:
        log.error(f" Failed to save transaction: {str(e)}")
        return False

@st.cache_data(ttl=300)  
def get_transactions_data() -> pd.DataFrame:
    """
    Fetch and process all transactions from Google Sheet.
    
    Returns:
        pd.DataFrame: Processed transactions data
    """
    try:
        log.debug("Fetching transactions data from Google Sheets")
        result: Any = service.spreadsheets().values().get( # type: ignore
            spreadsheetId=SHEET_ID,
            range='Expenses!A1:F'
        ).execute()
        
        values: list[list[str]] = result.get('values', [])
        if not values:
            log.warning("No transaction data found in sheet")
            return pd.DataFrame(columns=['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description'])
        
        log.info(f"Retrieved {len(values)-1} transaction records")
        return pd.DataFrame(values[1:], columns=['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description'])
    except Exception as e:
        log.error(f" Failed to fetch transactions data: {str(e)}")
        raise

def validate_amount(amount_str: str) -> float:
    """
    Validate and convert amount string to float.
    
    Args:
        amount_str: String representation of amount
        
    Returns:
        float: Validated amount
        
    Raises:
        ValueError: If amount is invalid
    """
    try:
        amount = float(amount_str)
        if amount <= 0:
            raise ValueError("Amount must be positive")
        return amount
    except ValueError as e:
        log.error(f"Invalid amount: {amount_str}")
        raise ValueError(f"Invalid amount: {amount_str}") from e

def classify_transaction_type(text: str, model: Any) -> dict[str, Any]:
    """
    Use Gemini to classify the type of transaction.
    """
    try:
        log.info("🔍 Starting transaction classification")
        log.debug(f"Input text: {text}")
        
        chat = model.start_chat(history=[])
        prompt = f"""
        Classify this transaction: '{text}'
        
        VERY IMPORTANT CLASSIFICATION RULES:
        1. If text contains "received pending" or "got pending" or "collected pending":
           -> MUST classify as PENDING_RECEIVED
           Example: "received pending money of 1275" -> PENDING_RECEIVED
           Example: "got pending payment of 500" -> PENDING_RECEIVED
        
        2. If text indicates future receipt WITHOUT "pending":
           -> classify as PENDING_TO_RECEIVE
           Example: "will receive 1000 next week" -> PENDING_TO_RECEIVE
        
        3. If text indicates future payment:
           -> classify as PENDING_TO_PAY
           Example: "need to pay 500 tomorrow" -> PENDING_TO_PAY
        
        4. If text indicates immediate expense:
           -> classify as EXPENSE_NORMAL
           Example: "spent 100 on food" -> EXPENSE_NORMAL
        
        5. If text indicates immediate income WITHOUT "pending":
           -> classify as INCOME_NORMAL
           Example: "got salary 5000" -> INCOME_NORMAL
        
        IMPORTANT: For any text containing "received pending", "got pending", or "collected pending",
        you MUST classify it as PENDING_RECEIVED, regardless of other words in the text.
        
        Respond in this format ONLY:
        type: <PENDING_RECEIVED/PENDING_TO_RECEIVE/PENDING_TO_PAY/EXPENSE_NORMAL/INCOME_NORMAL>
        amount: <positive number only>
        description: <brief description>
        """
        
        log.debug("Sending classification prompt to Gemini")
        response = chat.send_message(prompt)
        lines = response.text.strip().split('\n')
        result: dict[str, Any] = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
        
        # Double-check classification for pending received
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in ['received pending', 'got pending', 'collected pending']):
            if result.get('type') != 'PENDING_RECEIVED':
                log.warning(f"Correcting misclassified pending received transaction: {result.get('type')} -> PENDING_RECEIVED")
                result['type'] = 'PENDING_RECEIVED'
        
        # Validate required fields
        required_fields = ['type', 'amount', 'description']
        missing_fields = [field for field in required_fields if not result.get(field)]
        if missing_fields:
            st.chat_message("assistant").markdown("I couldn’t figure out the transaction amount or details. Could you be more specific?") # type: ignore
            log.warning(f"Missing required fields: {missing_fields}")
            return {}  # Gracefully exit to avoid exception

            
        # Validate transaction type
        valid_types = ['EXPENSE_NORMAL', 'INCOME_NORMAL', 'PENDING_TO_RECEIVE', 
                      'PENDING_TO_PAY', 'PENDING_RECEIVED', 'PENDING_PAID']
        if result['type'] not in valid_types:
            raise ValueError(f"Invalid transaction type: {result['type']}")
            
        # Validate amount
        result['amount'] = str(validate_amount(result['amount']))
        
        log.info(f"Transaction classified as: {result.get('type', 'UNKNOWN')}")
        log.debug(f"Classification details: {result}")
        return result
    except Exception as e:
        log.error(f"Failed to classify transaction: {str(e)}")
        raise
    
def detect_user_intent(text: str, model: Any) -> str:
    """Detect if user message is casual or related to finance."""
    try:
        chat = model.start_chat(history=[])
        prompt = f"""
        Analyze the user message: \"{text}\"
        Respond with only one word:
        - "finance" if it's related to money, expenses, income, savings etc.
        - "stocks" — if it mentions company names, stock prices, investment, market trends
        - "advisor" — if it asks for financial advice, money-saving tips, retirement, planning, investing, mutual funds, SIP, FD
        - "casual" if it's just chatting, greetings, questions etc.
        
        """
        response = chat.send_message(prompt)
        return response.text.strip().lower()
    except Exception as e:
        log.warning(f"Intent detection failed: {str(e)}")
        return 'finance'  # default to finance if unsure

def generate_casual_reply(text: str, model: Any) -> str:
    """Generate a casual assistant-style reply to friendly messages."""
    try:
        chat = model.start_chat(history=[])
        prompt = f"""
        You are a helpful, friendly assistant. Reply to this message casually:
        \"{text}\"
        Keep it short, fun or friendly. Do NOT suggest finance or commands unless user asked.
        """
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        log.warning(f"Casual reply generation failed: {str(e)}")
        return "Hey! How can I help you today?"
    
def generate_financial_advice(text: str, model: Any) -> str:
    """
    Generate general financial advice using Gemini.
    """
    try:
        chat = model.start_chat(history=[])
        prompt = f"""
        You are a helpful financial advisor.

        User's question: "{text}"

        Provide clear, simple, and actionable advice for beginners.
        Give 2–3 practical steps if it's a generic query like “how to invest”.
        Keep your tone friendly and clear.
        """
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        log.error(f"⚠️ Financial advice generation failed: {str(e)}")
        return "Sorry, I couldn't provide advice right now. Please try again later."


def render_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_received_pending_transaction(amount: float, description: str) -> tuple[bool, dict[str, Any] | None]:
    """
    Handle a pending transaction that has been received.
    """
    try:
        if amount <= 0:
            raise ValueError("Amount must be positive")
            
        log.info(f"💫 Processing received pending transaction: amount={amount}")
        
        # First check if this transaction was already processed today
        log.debug("Checking for existing received transactions today")
        today = datetime.now().strftime('%Y-%m-%d')
        
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range='Expenses!A:F'
        ).execute()
        
        values = result.get('values', [])
        if values and len(values) > 1:  # Check if we have data beyond header
            for row in values[1:]:  # Skip header
                if (len(row) >= 6 and 
                    row[0] == today and 
                    abs(float(row[1]) - amount) < 0.01 and
                    row[2] == 'Income' and
                    row[3] == 'Other' and
                    row[4] == 'Pending Received' and
                    'received pending' in row[5].lower()):
                    log.warning("⚠️ This pending transaction was already processed today")
                    return False, None
        
        # Now check pending transactions
        log.debug("Searching for matching pending transaction")
        result = service.spreadsheets().values().get( # type: ignore
            spreadsheetId=SHEET_ID,
            range='Pending!A:G'
        ).execute()
        
        values = result.get('values', []) # type: ignore
        if not values:
            log.warning("No pending transactions found in sheet")
            return False, None
            
        # Skip header row and find matching pending transaction
        matching_rows: list[int] = []
        
        # Validate sheet structure
        if len(values[0]) < 7: # type: ignore
            log.error("Invalid sheet structure: missing required columns")
            return False, None
            
        # Start from index 1 to skip header row
        for i, row in enumerate(values[1:], start=1): # type: ignore
            try:
                if len(row) < 7: # type: ignore
                    log.warning(f"⚠️ Skipping row {i+1}: insufficient columns")
                    continue
                    
                row_amount = float(row[1]) # type: ignore
                if (abs(row_amount - amount) < 0.01 and  # Use small epsilon for float comparison
                    row[6] == 'Pending' and 
                    row[2] == 'To Receive'):
                    matching_rows.append(i)
                    log.debug(f"Found potential match at row {i+1}: amount={row_amount}")
            except (ValueError, IndexError) as e:
                log.warning(f"Error processing row {i+1}: {str(e)}")
                continue
        
        if len(matching_rows) > 1:
            log.warning(f"Multiple matching pending transactions found for amount {amount}")
            # Use the most recent transaction if multiple matches
            row_index: int = matching_rows[-1]
            log.info(f"Selected most recent match at row {row_index+1}")
        elif len(matching_rows) == 1:
            row_index = matching_rows[0]
            log.info(f"Found matching pending transaction at row {row_index+1}")
        else:
            log.warning(f"No matching pending transaction found for amount {amount}")
            return False, None
            
        # Update status to Received
        log.debug(f"Updating status to Received for row {row_index+1}")
        range_name = f'Pending!G{row_index + 1}'
        try:
            service.spreadsheets().values().update( # type: ignore
                spreadsheetId=SHEET_ID,
                range=range_name,
                valueInputOption='RAW',
                body={'values': [['Received']]}
            ).execute()
        except Exception as e:
            log.error(f"Failed to update pending transaction status: {str(e)}")
            return False, None
        
        # Get original transaction details
        original_row = values[row_index] # type: ignore
        original_date = original_row[0] # type: ignore
        original_description = original_row[4] if len(original_row) > 4 else '' # type: ignore
        
        # Create transaction info
        transaction_info = {
            'type': 'Income',
            'amount': str(amount),
            'category': 'Other',
            'subcategory': 'Pending Received',
            'description': f"Received pending payment ({original_date}): {original_description}",
            'date': today
        }
        
        # Add as new Income transaction
        log.debug("Creating new Income transaction")
        success = add_transaction_to_sheet(
            transaction_info['date'],
            amount,
            transaction_info['type'],
            transaction_info['category'],
            transaction_info['subcategory'],
            transaction_info['description']
        )
        
        if success:
            log.info("Successfully processed received pending transaction")
        else:
            log.error("Failed to create Income transaction")
        
        return success, transaction_info if success else None
        
    except Exception as e:
        log.error(f"Failed to handle received pending transaction: {str(e)}")
        return False, None

def process_user_input(text: str) -> dict[str, Any]:
    """
    Process natural language input to extract transaction details.
    """
    try:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
               
        intent = detect_user_intent(text, model)
        log.info(f"Detected user intent: {intent}")

        if intent == 'casual':
            reply = generate_casual_reply(text, model)
            st.chat_message("assistant").markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            return {}  
        
        elif intent == 'advisor':
            log.info("Generating general financial advice")
            advice = generate_financial_advice(text, model)
            st.chat_message("assistant").markdown(f"📘 {advice}")
            st.session_state.messages.append({"role": "assistant", "content": advice})
            return {}

        
            
        log.info("Starting transaction processing")
        log.debug(f"Processing input: {text}")
        
        # Detect if this is a stock market query
        if any(keyword in text.lower() for keyword in ["stock", "price", "market", "invest", "share"]):
            log.info("Detected stock/finance query")
            
            ticker = detect_stock_ticker(text)
            if not ticker:
                st.chat_message("assistant").markdown("❓ I couldn't find a stock symbol. Try: *What's the price of TCS or Reliance?*")
                return {}
            
            data = get_stock_data(ticker)
            if not data:
                st.chat_message("assistant").markdown("❌ Failed to fetch stock info. Try again later.")
                return {}
            
            st.session_state.stock_result = {
            "data": data,
            "ticker": ticker,
            "advice": generate_stock_advice(
                f"Suggest a short-term and long-term investment strategy for {data['name']} stock (ticker: {ticker}) based on current price {data['price']} and general trends. Keep it beginner-friendly.",
                model
                )
            }
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"📈 Stock insight for {data['name']} ({ticker}) saved!"
            })
            return {}
            
            # Display info
            st.chat_message("assistant").markdown(
                f"📈 **{data['name']} ({ticker})**\n"
                f"💵 Current Price: `{data['price']} {data['currency']}`\n"
                f"📊 P/E Ratio: `{data['pe_ratio']}` | EPS: `{data['eps']}`\n"
                f"🏦 Market Cap: `{data['market_cap']}`"
            )

            # Trend chart
            st.subheader("📉 30-Day Trend")
            fig = px.line(data['history'], y="Close", title=f"{ticker} Price History")
            st.plotly_chart(fig, use_container_width=True)

            # Gemini advice
            prompt = f"Suggest a short-term and long-term investment strategy for {data['name']} stock (ticker: {ticker}) based on current price {data['price']} and general trends. Keep it beginner-friendly."
            advice = generate_stock_advice(prompt, model)
            st.chat_message("assistant").markdown(f"💡 **Investment Strategy**:\n{advice}")
            
            return {}  # Do not run transaction logic

        
        # First, classify the transaction type
        log.debug("Step 1: Classifying transaction type")
        classification = classify_transaction_type(text, model)
        if not classification:
            return {}
        transaction_type = classification.get('type', '')
        
        try:
            amount = float(classification.get('amount', 0))
            if amount <= 0:
                raise ValueError("Amount must be positive")
        except ValueError as e:
                log.error(f"Invalid amount detected in classification: {classification.get('amount')}. Error: {e}")
                st.chat_message("assistant").markdown("Hmm, I couldn't understand the transaction properly. Could you try rephrasing it?")
                return {}
        
        log.info(f"Transaction classified as: {transaction_type}")
        
        # Handle each type differently
        if transaction_type == 'PENDING_RECEIVED':
            log.info("Handling received pending transaction")
            success, transaction_info = handle_received_pending_transaction(amount, text)
            if success and transaction_info:
                log.info("Successfully processed received pending transaction")
                # Mark transaction as auto-processed to skip form
                transaction_info['auto_processed'] = True
                return transaction_info
            else:
                log.warning("⚠️ Failed to process received pending transaction")
                raise ValueError("Failed to process received pending transaction")
        
        elif transaction_type == 'PENDING_PAID':
            log.info("💰 Handling paid pending transaction")
            # TODO: Implement handling paid pending payments
            raise NotImplementedError("Handling paid pending transactions is not implemented yet")
            
        # For other types, get detailed transaction info
        log.debug("Step 2: Getting detailed transaction info")
        chat = model.start_chat(history=[])
        prompt = f"""
        Extract transaction information from this text: '{text}'
        Transaction was classified as: {transaction_type}
        
        Based on the classification, follow these rules:
        
        1. For EXPENSE_NORMAL:
           -> Set type: "Expense"
           -> Choose category from: Food/Transportation/Housing/Entertainment/Shopping/Healthcare/Gift/Other
        
        2. For INCOME_NORMAL:
           -> Set type: "Income"
           -> Choose category from: Salary/Investment/Other
        
        3. For PENDING_TO_RECEIVE:
           -> Set type: "To Receive"
           -> Set category: "Pending Income"
        
        4. For PENDING_TO_PAY:
           -> Set type: "To Pay"
           -> Choose category from: Bills/Debt
        
        5. For PENDING_RECEIVED:
           -> Set type: "Income"
           -> Set category: "Other"
           -> Set subcategory: "Pending Received"
        
        6. For PENDING_PAID:
           -> Set type: "Expense"
           -> Use original pending payment category
        
        Respond in this EXACT format (include ALL fields):
        type: <Income/Expense/To Receive/To Pay>
        amount: <number only>
        category: <must match categories listed above>
        subcategory: <must match valid subcategories>
        description: <brief description>
        due_date: <YYYY-MM-DD format, ONLY for To Receive/To Pay>
        """
        
        log.debug("🤖 Sending detail extraction prompt to Gemini")
        response = chat.send_message(prompt)
        response_text: str = response.text
        lines: list[str] = response_text.strip().split('\n')
        extracted_info: dict[str, Any] = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                extracted_info[key.strip()] = value.strip().replace('"', '').replace("'", "")
        
        log.debug(f"Extracted transaction details: {extracted_info}")
        
        # Set current date as transaction date
        current_date: str = datetime.now().strftime('%Y-%m-%d')
        extracted_info['date'] = current_date
        
        # Handle relative dates in due_date
        if extracted_info.get('type') in ['To Receive', 'To Pay']:
            log.debug("Processing due date for pending transaction")
            if 'due_date' not in extracted_info or not extracted_info.get('due_date', '').strip():
                due_date: str = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                extracted_info['due_date'] = due_date
                log.debug(f"No due date provided, defaulting to: {due_date}")
            else:
                try:
                    parsed_date: datetime = parser.parse(str(extracted_info.get('due_date', '')))
                    extracted_info['due_date'] = parsed_date.strftime('%Y-%m-%d')
                    log.debug(f"Parsed due date: {extracted_info['due_date']}")
                except:
                    due_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                    extracted_info['due_date'] = due_date
                    log.warning(f"Failed to parse due date, defaulting to: {due_date}")
        
        log.info("Successfully processed transaction")
        log.debug(f"Final transaction info: {extracted_info}")
        return extracted_info
        
    except Exception as e:
        log.error(f"Failed to process user input: {str(e)}", exc_info=True)
        raise




def show_analytics() -> None:
    """
    Display analytics dashboard with transaction visualizations.
    Shows pie charts and trends for income and expenses.
    """
    try:
        log.info("Generating financial analytics")
        df = get_transactions_data()
        
        if df.empty:
            st.info("No transactions recorded yet. Add some transactions to see analytics!")
            return
            
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce') # type: ignore
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # type: ignore
        
        # Calculate totals
        total_income = df[df['Type'] == 'Income']['Amount'].sum() # type: ignore 
        total_expenses = df[df['Type'] == 'Expense']['Amount'].sum() # type: ignore
        net_balance = total_income - total_expenses
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Income", f"Rs. {total_income:,.2f}", delta=None)
        with col2:
            st.metric("Total Expenses", f"Rs. {total_expenses:,.2f}", delta=None)
        with col3:
            st.metric("Net Balance", f"Rs. {net_balance:,.2f}", 
                     delta=f"Rs. {net_balance:,.2f}", 
                     delta_color="normal" if net_balance >= 0 else "inverse")
        
        if len(df) > 1:  # Only show charts if we have more than one transaction
            # Income vs Expenses over time
            df_grouped = df.groupby(['Date', 'Type'])['Amount'].sum().unstack(fill_value=0) # type: ignore
            fig_timeline = px.line(df_grouped,  # type: ignore
                                 title='Income vs Expenses Over Time',
                                 labels={'value': 'Amount (Rs. )', 'variable': 'Type'})
            
            fig_timeline.update_layout( # type: ignore
                plot_bgcolor='#fffbe6',
                paper_bgcolor='#fffbe6',
                font=dict(color='#333', size=14),
                margin=dict(t=30, b=30, l=10, r=10),
            )
            st.plotly_chart(fig_timeline) # type: ignore
            
            
            
            # Category breakdown for both income and expenses
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Income Breakdown")
                income_df = df[df['Type'] == 'Income']
                if not income_df.empty:
                    fig_income = px.pie(income_df, values='Amount', names='Category',  # type: ignore
                                      title='Income by Category')
                    fig_income.update_layout( # type: ignore
                        paper_bgcolor='#fffbe6',
                        font=dict(color='#333', size=14),
                        margin=dict(t=30, b=30, l=10, r=10),
                    )
                    st.plotly_chart(fig_income) # type: ignore
                else:
                    st.info("No income transactions recorded yet.")
            
            with col2:
                st.subheader("Expense Breakdown")
                expense_df = df[df['Type'] == 'Expense']
                if not expense_df.empty:
                    fig_expense = px.pie( # type: ignore
                        expense_df,
                        values='Amount',
                        names='Category',
                        title='Expenses by Category',
                        color_discrete_sequence=[
                            '#f39c12', '#e67e22', '#d35400', '#f7c59f', '#f8b195',
                            '#f67280', '#c06c84', '#6c5b7b'  # vibrant theme
                        ]
                    )
                    fig_expense.update_layout( # type: ignore
                        paper_bgcolor='#fffbe6',
                        font=dict(color='#333', size=14),
                        margin=dict(t=30, b=30, l=10, r=10),
                    )
                    st.plotly_chart(fig_expense) # type: ignore

                else:
                    st.info("No expense transactions recorded yet.")
            
            # Monthly summary
            st.subheader("Monthly Summary")
            monthly_summary = df.groupby([df['Date'].dt.strftime('%Y-%m'), 'Type'])['Amount'].sum().unstack(fill_value=0) # type: ignore
            monthly_summary['Net'] = monthly_summary.get('Income', 0) - monthly_summary.get('Expense', 0) # type: ignore
            st.dataframe(monthly_summary.style.format("Rs. {:,.2f}")) # type: ignore
        
        log.info("✅ Analytics visualizations generated successfully")
    except Exception as e:
        log.error(f"❌ Failed to generate analytics: {str(e)}")
        st.error("Failed to generate analytics. Please try again later.")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_sheet_url() -> str:
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"

@st.cache_resource  # Cache for the entire session
def initialize_gemini() -> Any:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY')) # type: ignore
    return genai.GenerativeModel('gemini-1.5-flash') # type: ignore

@st.cache_data
def get_subcategories(trans_type: str, category: str) -> list[str]:
    return CATEGORIES[trans_type][category]

def on_save_click():
    st.session_state.save_clicked = True

def verify_sheet_setup() -> bool:
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range='Expenses!A1:F1'
        ).execute()
        
        values = result.get('values', [])
        expected_headers = ['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description']
        
        if not values or values[0] != expected_headers:
            # Reinitialize headers
            headers = [expected_headers]
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range='Expenses!A1:F1',
                valueInputOption='RAW',
                body={'values': headers}
            ).execute()
            log.info("Headers reinitialized")
            
        return True
    except Exception as e:
        log.error(f"Failed to verify sheet setup: {str(e)}")
        return False

def show_success_message(transaction_date: datetime | str, subcategory: str | None) -> None:
    """
    Display success message after transaction is saved.
    
    Args:
        transaction_date: Date of the transaction
        subcategory: Transaction subcategory, if applicable
    """
    emoji = "💰" if st.session_state.current_transaction['type'] == "Income" else "💸"
    confirmation_message = (
        f"{emoji} Transaction recorded:\n\n"
        f"Date: {transaction_date}\n"
        f"Amount: Rs. {float(st.session_state.current_transaction['amount']):,.2f}\n"
        f"Type: {st.session_state.current_transaction['type']}\n"
        f"Category: {st.session_state.current_transaction['category']}\n"
        f"Subcategory: {subcategory if subcategory else 'N/A'}"
    )
    st.success(confirmation_message)
    st.session_state.messages.append({"role": "assistant", "content": confirmation_message})
    log.info("✅ Transaction saved and analytics updated")

def show_transaction_form():
    """Separate function to handle transaction form display and processing"""
    extracted_info = st.session_state.current_transaction
    
    # Skip form for auto-processed transactions (like received pending)
    if extracted_info.get('auto_processed'):
        log.debug("Showing feedback for auto-processed transaction")
        
        # Show detailed success message
        st.success("✅ Transaction Processed Successfully")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Transaction Details:**")
            st.write(f"📅 Date: {extracted_info.get('date')}")
            st.write(f"💰 Amount: Rs. {float(extracted_info.get('amount', 0)):,.2f}")
            st.write(f"📝 Type: {extracted_info.get('type')}")
            
        with col2:
            st.write(f"🏷️ Category: {extracted_info.get('category')}")
            st.write(f"📑 Subcategory: {extracted_info.get('subcategory')}")
            st.write(f"📌 Description: {extracted_info.get('description')}")
            
        # Add a divider for visual separation
        st.divider()
        
        # Add a clear button
        if st.button("Clear Message", key="clear_feedback"):
            st.session_state.current_transaction = None
            st.rerun()
        return
    
    if 'amount' in extracted_info and 'type' in extracted_info:
        # Create form container
        form_container = st.container()
        
        with form_container:
            # Initialize form state
            if 'form_submitted' not in st.session_state:
                st.session_state.form_submitted = False
            
            with st.form(key="transaction_form"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if extracted_info['type'] in ['To Receive', 'To Pay']:
                        # For pending transactions
                        try:
                            # Try to parse the due date if it exists
                            if 'due_date' in extracted_info and extracted_info['due_date']:
                                default_due_date = datetime.strptime(extracted_info['due_date'], '%Y-%m-%d')
                            else:
                                # Default to 7 days from now
                                default_due_date = datetime.now() + timedelta(days=7)
                        except ValueError:
                            # If parsing fails, use 7 days from now
                            default_due_date = datetime.now() + timedelta(days=7)
                        
                        due_date = st.date_input(
                            "Due date",
                            value=default_due_date,
                            key="due_date"
                        )
                    else:
                        # For regular transactions
                        categories = get_categories()
                        subcategories = categories[extracted_info['type']][extracted_info['category']]
                        subcategory = st.selectbox(
                            "Select subcategory",
                            subcategories,
                            key="subcategory_select"
                        )
                    
                    default_date = datetime.strptime(extracted_info['date'], '%Y-%m-%d')
                    transaction_date = st.date_input(
                        "Transaction date",
                        value=default_date,
                        key="transaction_date"
                    )
                
                with col2:
                    submitted = st.form_submit_button(
                        "Save",
                        type="primary",
                        use_container_width=True,
                        on_click=lambda: setattr(st.session_state, 'form_submitted', True)
                    )

            if st.session_state.form_submitted:
                try:
                    if extracted_info['type'] in ['To Receive', 'To Pay']:
                        success = add_pending_transaction_to_sheet(
                            transaction_date.strftime('%Y-%m-%d'),  # Convert to string
                            extracted_info['amount'],
                            extracted_info['type'],
                            extracted_info['category'],
                            extracted_info.get('description', ''),
                            due_date.strftime('%Y-%m-%d')  # Convert to string
                        )
                    else:
                        success = add_transaction_to_sheet(
                            transaction_date.strftime('%Y-%m-%d'),  # Convert to string
                            extracted_info['amount'],
                            extracted_info['type'],
                            extracted_info['category'],
                            subcategory,
                            extracted_info.get('description', '')
                        )
                    
                    if success:
                        show_success_message(
                            transaction_date.strftime('%Y-%m-%d'),  # Convert to string
                            subcategory if 'subcategory' in locals() else None
                        )
                        st.session_state.current_transaction = None
                        st.session_state.form_submitted = False
                        st.rerun()
                    else:
                        st.error("Failed to save transaction. Please try again.")
                        st.session_state.form_submitted = False
                except Exception as e:
                    log.error(f"Failed to save transaction: {str(e)}")
                    st.error("An error occurred while saving the transaction. Please try again.")
                    st.session_state.form_submitted = False

def add_pending_transaction_to_sheet(date, amount, trans_type, category, description, due_date):
    try:
        # Verify sheets exist before adding transaction
        if not verify_sheets_setup():
            raise Exception("Failed to verify sheets setup")
            
        log.info(f"Starting pending transaction save: {date}, {amount}, {trans_type}, {category}, {description}, {due_date}")
        
        # Format the dates if they're datetime objects
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        if isinstance(due_date, datetime):
            due_date = due_date.strftime('%Y-%m-%d')
        
        # Ensure amount is a string
        amount = str(float(amount))
        
        # Prepare the values with initial status as 'Pending'
        values = [[str(date), amount, trans_type, category, description, str(due_date), 'Pending']]
        
        result = service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range='Pending!A1:G1',
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body={'values': values}
        ).execute()
        
        log.info(f"Pending transaction saved successfully: {result}")
        return True
        
    except Exception as e:
        log.error(f"Failed to save pending transaction: {str(e)}")
        return False

def verify_sheets_setup():
    """Verify both Expenses and Pending sheets exist with correct headers"""
    try:
        # Get all sheets
        sheet_metadata = service.spreadsheets().get(spreadsheetId=SHEET_ID).execute()
        sheets = sheet_metadata.get('sheets', '')
        existing_sheets = {s.get("properties", {}).get("title") for s in sheets}
        
        # Check and initialize Expenses sheet
        if 'Expenses' not in existing_sheets:
            log.info("Creating new Expenses sheet...")
            body = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': 'Expenses'
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=SHEET_ID,
                body=body
            ).execute()
            
            headers = [['Date', 'Amount', 'Type', 'Category', 'Subcategory', 'Description']]
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range='Expenses!A1:F1',
                valueInputOption='RAW',
                body={'values': headers}
            ).execute()
        
        # Check and initialize Pending sheet
        if 'Pending' not in existing_sheets:
            log.info("Creating new Pending sheet...")
            body = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': 'Pending'
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=SHEET_ID,
                body=body
            ).execute()
            
            headers = [['Date', 'Amount', 'Type', 'Category', 'Description', 'Due Date', 'Status']]
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range='Pending!A1:G1',
                valueInputOption='RAW',
                body={'values': headers}
            ).execute()
            
        log.info("✨ Sheets verified and initialized")
        return True
    except Exception as e:
        log.error(f" Failed to verify/initialize sheets: {str(e)}")
        return False

def main():
    """
    Main application function.
    Handles the core application flow and user interface.
    """
    try:
        log.info("🚀 Starting Finance Tracker application")
        
        # Initialize session state
        if 'sheets_verified' not in st.session_state:
            st.session_state.sheets_verified = False
        
        # Only verify sheets once
        if not st.session_state.sheets_verified:
            verify_sheets_setup()
            st.session_state.sheets_verified = True
        
        st.title("💰 Smart Finance Tracker")
        st.markdown(f"📊 [View Google Sheet]({get_sheet_url()})")
        st.divider()
        
        init_session_state()
        
        # Show analytics if selected
        if st.session_state.show_analytics:
            show_analytics()
            return

    # Chat + Form UI
        render_chat()
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Handle chat input
        if prompt := st.chat_input("Tell me about your income or expense..."):
            log.debug(f"Received user input: {prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process user input only if we don't have a current transaction
            if not st.session_state.current_transaction:
                try:
                    extracted_info = process_user_input(prompt)
                except Exception as e:
                    log.error(f"Processing failed: {str(e)}", exc_info=True)
                    st.chat_message("assistant").markdown("😕 Specify your expenses?")
                    return
                                
                st.session_state.current_transaction = extracted_info
                st.rerun()
            
        # Show transaction form if we have extracted info
        if st.session_state.current_transaction:
            show_transaction_form()
            

        # ✅ Now show stock result if available
        if st.session_state.get('stock_result'):
            stock = st.session_state.stock_result["data"]
            ticker = st.session_state.stock_result["ticker"]
            advice = st.session_state.stock_result["advice"]

            st.subheader(f"📊 Stock Details: {stock['name']} ({ticker})")
            st.markdown(f"""
                - 💵 **Price**: `{stock['price']} {stock['currency']}`
                - 📊 **P/E**: `{stock['pe_ratio']}` | **EPS**: `{stock['eps']}`
                - 🏦 **Market Cap**: `{stock['market_cap']}`
            """)
            fig = px.line(stock['history'], y="Close", title=f"{ticker} Price Trend")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"💡 **Advice**: {advice}")
            
            if st.button("Clear Stock Info", key="clear_stock"):
                st.session_state.stock_result = None
                st.rerun()

    
    except Exception as e:
        log.error(f"❌ Application error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()
