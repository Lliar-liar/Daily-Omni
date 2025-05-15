# config_tester.py

import platform
import os

# --- Model Selection ---
# Options: 'gemini_av', 'gemini_visual', 'gpt4o_visual', 'gpt4o_text', 'deepseek_text'
DEFAULT_MODEL_TYPE = 'gemini_av'

# --- Execution Mode ---
# Options: 'sequential', 'parallel'
DEFAULT_EXECUTION_MODE = 'parallel'

# --- Paths ---
SYSTEM_OS = platform.system()

BASE_VIDEO_DIR = "./Videos"
FFMPEG_PATH = 'ffmpeg' 
DEFAULT_QA_JSON_PATH = 'qa.json'

# --- API Keys & Endpoints ---
# ** IMPORTANT: Use environment variables or a secure method for production **
GEMINI_API_KEY_1 = os.environ.get('GEMINI_API_KEY_1', 'YOUR_API_KEY') # Key from test_gemini_visual
GEMINI_API_KEY_2 = os.environ.get('GEMINI_API_KEY_2', 'YOUR_API_KEY') # Key from test_gemini_av (Using distinct names for clarity)

# Use API Key 2 for Gemini models by default, as it was in the AV script
GEMINI_API_KEY = GEMINI_API_KEY_2

# OpenAI Compatible Endpoints (GPT-4o / DeepSeek)
# GPT-4o via ChatAnywhere
GPT4O_API_KEY = os.environ.get('GPT4O_API_KEY', 'YOUR_API_KEY')
GPT4O_BASE_URL = os.environ.get('GPT4O_BASE_URL', 'YOUR_BASE_URL')
GPT4O_MODEL_NAME = "gpt-4o-ca"

# DeepSeek
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'YOUR_API_KEY')
DEEPSEEK_BASE_URL = os.environ.get('DEEPSEEK_BASE_URL', 'YOUR_BASE_URL')
DEEPSEEK_MODEL_NAME = "deepseek-chat"

# --- Model Specific Settings ---
GEMINI_AV_MODEL_NAME = 'gemini-2.0-flash'
GEMINI_VISUAL_MODEL_NAME = 'gemini-2.0-flash' # Can be the same or different

# Settings for frame extraction (GPT-4o Visual)
SECONDS_PER_FRAME_GPT4O = 2

# --- API Call Settings ---
MAX_RETRIES = 4
BASE_DELAY = 4 # seconds

# --- Parallel Processing ---
# Use None to default to os.cpu_count() or set a specific number
MAX_WORKERS = None

# --- Gemini Safety Settings ---
# Block harmful content minimally for testing, adjust as needed
GEMINI_SAFETY_SETTINGS = {
    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
}