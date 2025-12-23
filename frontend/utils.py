import streamlit as st
import requests
from typing import Dict, Any, Optional

API_BASE = "http://127.0.0.1:8000"

def setup_page(title: str, icon: str = "📊"):
    """Professional page setup"""
    st.markdown(f"# {icon} **{title}**")
    st.markdown("---")

def make_get_request(endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
    """Safe API GET request"""
    try:
        response = requests.get(f"{API_BASE}/{endpoint}", params=params, timeout=30)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"⚠️ API error: {str(e)}")
        return None

def make_post_request(endpoint: str, json_data: Dict[str, Any] = None) -> Optional[Dict]:
    """Safe API POST request"""
    try:
        response = requests.post(f"{API_BASE}/{endpoint}", json=json_data, timeout=60)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"⚠️ API error: {str(e)}")
        return None
