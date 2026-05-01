from __future__ import annotations

import os
import requests
import urllib3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def call_uc3m_api(prompt: str, model_name: str = "llama3.1:8b") -> str:
    api_url = "https://yiyuan.tsc.uc3m.es/api/generate"
    
    # Retrieve the key loaded by load_dotenv()
    api_key = os.environ.get("UC3M_API_KEY")
    
    if not api_key:
        return "Error: UC3M_API_KEY is not set. Please check your .env file."

    headers = {"X-API-KEY": api_key}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    try:
        # verify=False is kept as per your requirement for the UC3M environment
        response = requests.post(api_url, headers=headers, json=payload, verify=False, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "Error: No response field.")
    except Exception as e:
        return f"API connection failed: {str(e)}"