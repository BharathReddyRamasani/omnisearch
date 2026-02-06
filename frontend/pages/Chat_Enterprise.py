"""
ENTERPRISE CHAT UI
==================
Streamlit interface for enterprise chat with DSL transparency.

FEATURES:
✅ Natural language queries
✅ DSL transparency (shows what query was executed)
✅ Clarification UI (when ambiguous)
✅ Confidence scoring
✅ Audit trail
❌ No raw data exposure
❌ No hallucination
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import uuid
import pandas as pd

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="OmniSearch Enterprise Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# STYLING
# ============================================
st.markdown("""
<style>
    .chat-header {
        background: linear-gradient(135deg, #0052cc 0%, #1e6ed4 50%, #2563eb 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(5, 82, 204, 0.2);
    }
    .dsl-panel {
        background: #f5f5f5;
        border-left: 4px solid #0052cc;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .confidence-high {
        color: #10b981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f59e0b;
        font-weight: bold;
    }
    .confidence-low {
        color: #ef4444;
        font-weight: bold;
    }
    .clarification-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-box {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-box {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="chat-header">
    <h1 style="margin: 0;">🤖 OmniSearch Enterprise Chat</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.95;">
        Intelligent analytics with zero hallucination guarantee
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - DATASET SELECTION
# ============================================
st.sidebar.markdown("### 📊 Dataset Selection")

# Get available datasets from backend
try:
    response = requests.get(f"{API}/datasets/list", timeout=5)
    if response.status_code == 200:
        datasets = response.json().get('datasets', [])
        dataset_names = [d['id'] for d in datasets]
    else:
        dataset_names = ['00f53e8a']  # Fallback
except:
    dataset_names = ['00f53e8a']

selected_dataset = st.sidebar.selectbox(
    "Choose a dataset:",
    dataset_names,
    help="Select the dataset to query"
)

# Store in session
if 'dataset_id' not in st.session_state:
    st.session_state.dataset_id = selected_dataset
else:
    st.session_state.dataset_id = selected_dataset

# ============================================
# SIDEBAR - SETTINGS
# ============================================
st.sidebar.markdown("### ⚙️ Settings")

show_dsl = st.sidebar.checkbox(
    "Show DSL Query Details",
    value=True,
    help="Display the generated JSON DSL"
)

show_confidence = st.sidebar.checkbox(
    "Show Confidence Score",
    value=True,
    help="Display confidence in the result"
)

show_audit_id = st.sidebar.checkbox(
    "Show Audit ID",
    value=False,
    help="Display request ID for audit trail"
)

# ============================================
# MAIN CONTENT
# ============================================

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'clarification_context' not in st.session_state:
    st.session_state.clarification_context = None

# Chat message display
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.info("""
        👋 Welcome to OmniSearch Enterprise Chat!
        
        **Ask questions about your data:**
        - "Tell me about this dataset"
        - "What is the average price?"
        - "How do sales vary by region?"
        - "Show me the distribution of customer age"
        
        💡 **How it works:**
        1. Your natural language query is converted to a strict JSON DSL
        2. The DSL is validated against the dataset schema
        3. If unclear, I'll ask for clarification
        4. The validated query executes safely
        5. Results are explained in business language
        
        🔒 **Safety Guarantees:**
        - No hallucination (DSL validates every query)
        - No data invention (columns must exist)
        - No unsafe operations (only aggregate, group, correlate, etc.)
        - Full audit trail (every query logged)
        """)
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg.get('role', 'user')):
                st.write(msg.get('content', ''))
                
                # Display DSL if bot message and show_dsl enabled
                if msg.get('role') == 'assistant' and show_dsl and 'dsl' in msg:
                    with st.expander("📋 Query Details (DSL)"):
                        st.json(msg['dsl'])
                
                # Display confidence if available
                if msg.get('role') == 'assistant' and show_confidence and 'confidence' in msg:
                    confidence_class = f"confidence-{msg['confidence']}"
                    st.markdown(
                        f"**Confidence:** <span class='{confidence_class}'>"
                        f"{msg['confidence'].upper()}</span>",
                        unsafe_allow_html=True
                    )
                
                # Display audit ID if requested
                if msg.get('role') == 'assistant' and show_audit_id and 'audit_id' in msg:
                    st.caption(f"🔍 Audit ID: {msg['audit_id']}")

# ============================================
# CLARIFICATION HANDLING
# ============================================

if st.session_state.clarification_context:
    clarification = st.session_state.clarification_context
    
    st.markdown('<div class="clarification-box">', unsafe_allow_html=True)
    st.markdown(f"**❓ {clarification.get('question')}**")
    
    options = clarification.get('options', [])
    
    if options:
        choice_idx = st.radio(
            "Select an option:",
            range(len(options)),
            format_func=lambda i: options[i].get('interpretation', f'Option {i+1}'),
            key="clarification_choice"
        )
        
        if st.button("✓ Confirm", key="confirm_clarification"):
            # Send confirmation to API
            try:
                response = requests.post(
                    f"{API}/chat/clarification",
                    json={
                        'dataset_id': st.session_state.dataset_id,
                        'choice_index': choice_idx,
                        'clarification_context': clarification
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add assistant response
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': result.get('response', 'Query executed'),
                        'dsl': result.get('dsl'),
                        'confidence': result.get('confidence', 'medium'),
                        'audit_id': result.get('audit_id')
                    })
                    
                    # Clear clarification context
                    st.session_state.clarification_context = None
                    st.rerun()
                else:
                    st.error(f"Error: {response.text}")
            
            except requests.exceptions.Timeout:
                st.error("Request timeout. Please try again.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# INPUT & QUERY HANDLING
# ============================================

# Chat input
user_input = st.chat_input(
    "Ask a question about your data...",
    key="chat_input"
)

if user_input:
    # Add user message to chat
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input
    })
    
    # Send to backend
    with st.spinner("🔄 Processing your query..."):
        try:
            response = requests.post(
                f"{API}/chat",
                json={
                    'dataset_id': st.session_state.dataset_id,
                    'query': user_input,
                    'history': st.session_state.messages[:-1]  # Exclude current message
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                status = result.get('status')
                
                if status == 'clarification_needed':
                    # Store clarification context
                    st.session_state.clarification_context = result.get('clarification_context')
                    
                    # Add bot message
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': result.get('response', 'Please clarify your intent'),
                        'audit_id': result.get('audit_id')
                    })
                
                elif status == 'ok':
                    # Add bot message with details
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': result.get('response', 'Query executed successfully'),
                        'dsl': result.get('dsl'),
                        'result': result.get('result'),
                        'confidence': result.get('confidence', 'medium'),
                        'audit_id': result.get('audit_id'),
                        'warnings': result.get('warnings')
                    })
                
                else:  # error
                    st.error(f"❌ Error: {result.get('response', 'Unknown error')}")
                
                st.rerun()
            
            else:
                st.error(f"Server error: {response.status_code}")
                st.error(response.text)
        
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timeout. Please try again.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================
# FOOTER - INFO
# ============================================
st.markdown("---")
st.markdown("""
### 📚 Documentation

**DSL Actions Available:**
- `describe` - Get dataset overview
- `aggregate` - Calculate metrics (mean, sum, count, etc.)
- `groupby` - Group by categorical column and aggregate
- `correlation` - Find relationships between columns
- `distribution` - Analyze value distribution
- `model_info` - View trained model metadata

**Safety Constraints:**
- Maximum 4 columns per query
- Maximum 3 groupby columns
- Metrics must match column types
- Columns must exist in dataset

**Confidence Levels:**
- 🟢 **HIGH**: Clear intent, sufficient data
- 🟡 **MEDIUM**: Some ambiguity or limited data
- 🔴 **LOW**: High uncertainty or edge cases
""")
