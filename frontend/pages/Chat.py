import streamlit as st
import requests
import json
import time
from datetime import datetime
import uuid
import pandas as pd

API = "http://127.0.0.1:8003"

st.set_page_config(
    page_title="OmniSearch AI – Intelligent Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# INDUSTRIAL CSS STYLING
# =====================================================
st.markdown("""
<style>
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .chat-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .chat-subtitle {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        max-height: 600px;
        overflow-y: auto;
    }
    .message-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .message-bot {
        background: #f8f9fa;
        color: #333;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .message-timestamp {
        font-size: 0.8rem;
        opacity: 0.7;
        margin-top: 0.5rem;
    }
    .input-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .typing-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .suggestion-chip {
        display: inline-block;
        background: #e9ecef;
        color: #495057;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        cursor: pointer;
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }
    .suggestion-chip:hover {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
    }
    .sidebar-chat-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .conversation-summary {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("### 🤖 **AI Assistant**")
    st.markdown("---")

    # Chat statistics
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    total_messages = len(st.session_state.chat_history)
    user_messages = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
    bot_messages = len([m for m in st.session_state.chat_history if m['role'] == 'assistant'])

    st.markdown(f"""
    <div class="sidebar-chat-info">
        <h4>📊 Session Stats</h4>
        <p><strong>Total Messages:</strong> {total_messages}</p>
        <p><strong>Your Messages:</strong> {user_messages}</p>
        <p><strong>AI Responses:</strong> {bot_messages}</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick actions
    st.markdown("### ⚡ **Quick Actions**")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("📄 Export Conversation", use_container_width=True):
        if st.session_state.chat_history:
            conversation_text = "\n\n".join([
                f"{'You' if msg['role'] == 'user' else 'AI'}: {msg['content']}\n{msg['timestamp']}"
                for msg in st.session_state.chat_history
            ])
            st.download_button(
                "Download Chat Log",
                conversation_text,
                file_name=f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    # Conversation insights
    if st.session_state.chat_history:
        topics = []
        for msg in st.session_state.chat_history:
            content = msg['content'].lower()
            if any(word in content for word in ['accuracy', 'score', 'performance']):
                topics.append('Model Performance')
            if any(word in content for word in ['feature', 'column', 'variable']):
                topics.append('Feature Analysis')
            if any(word in content for word in ['predict', 'forecast', 'estimate']):
                topics.append('Predictions')

        if topics:
            st.markdown(f"""
            <div class="conversation-summary">
                <h4>🎯 Discussion Topics</h4>
                <p>{', '.join(set(topics))}</p>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# MAIN HEADER
# =====================================================
st.markdown("""
<div class="chat-header">
    <h1 class="chat-title">🤖 Intelligent ML Assistant</h1>
    <p class="chat-subtitle">Ask questions about your data, models, and insights • Powered by Advanced AI</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SESSION CHECK
# =====================================================
if 'dataset_id' not in st.session_state:
    st.warning("⚠️ **No Dataset Loaded**")
    st.info("Please upload a dataset first to start chatting with your data.")
    st.stop()

dataset_id = st.session_state.dataset_id

# =====================================================
# CHAT QUERY PROCESSOR
# =====================================================
def process_chat_query(question, dataset_id):
    """
    Industrial-level chat processing with comprehensive AI understanding
    """
    try:
        payload = {
            "question": question,
            "history": st.session_state.get('chat_history', [])[-10:]  # Last 10 messages for context
        }
        resp = requests.post(f"{API}/api/chat/{dataset_id}", json=payload, timeout=60)  # Increased timeout
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "ok":
                return {
                    'text_response': data.get("answer", "No answer generated"),
                    'structured_data': data.get("structured_data"),
                    'context_used': data.get("context_used", [])
                }
            else:
                return {
                    'text_response': f"Error: {data.get('message', 'Unknown error')}",
                    'structured_data': None
                }
        else:
            return {
                'text_response': f"Backend error: {resp.status_code} - {resp.text}",
                'structured_data': None
            }
    except Exception as e:
        return {
            'text_response': f"Connection error: {str(e)}",
            'structured_data': None
        }

# =====================================================
# CHAT CONTAINER
# =====================================================
st.markdown("### 💬 **Conversation**")

# Initialize chat history if not exists
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #6c757d;">
            <h3>👋 Welcome to your AI Data Assistant!</h3>
            <p>Start a conversation about your dataset, models, or ask for insights.</p>
            <p><em>Try asking: "What's the model accuracy?" or "Show me top features"</em></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="message-user">
                    <strong>You:</strong> {message['content']}
                    <div class="message-timestamp">{message['timestamp']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Check if message has structured data
                if 'structured_data' in message:
                    st.markdown(f"""
                    <div class="message-bot">
                        <strong>AI Assistant:</strong> {message['content']}
                        <div class="message-timestamp">{message['timestamp']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display structured data
                    if message['structured_data'].get('type') == 'dataframe':
                        st.dataframe(message['structured_data']['data'])
                    elif message['structured_data'].get('type') == 'metrics':
                        cols = st.columns(len(message['structured_data']['data']))
                        for i, (key, value) in enumerate(message['structured_data']['data'].items()):
                            with cols[i % len(cols)]:
                                st.metric(key.replace('_', ' ').title(), value)
                    elif message['structured_data'].get('type') == 'chart':
                        # This would be handled by the backend returning chart data
                        pass
                else:
                    st.markdown(f"""
                    <div class="message-bot">
                        <strong>AI Assistant:</strong> {message['content']}
                        <div class="message-timestamp">{message['timestamp']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# =====================================================
# SUGGESTED QUESTIONS
# =====================================================
if not st.session_state.chat_history:
    st.markdown("### 💡 **Suggested Questions**")

    suggestions = [
        "What's the accuracy of my trained model?",
        "Show me the top features for prediction",
        "How many missing values are in my dataset?",
        "What type of ML task is this?",
        "Compare the performance of different models",
        "What are the most important variables?",
        "How clean is my data?",
        "What predictions can I make?"
    ]

    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.user_input = suggestion
                st.rerun()

# =====================================================
# INPUT SECTION
# =====================================================
st.markdown("### ✍️ **Ask Your Question**")

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Type your question here...",
        height=100,
        placeholder="Ask me anything about your data, models, or insights...",
        help="Be specific for better answers. Try questions like 'What's the model performance?' or 'Show me data quality metrics'"
    )

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        submit_button = st.form_submit_button(
            "🚀 Send Message",
            type="primary",
            use_container_width=True
        )

    with col2:
        if st.form_submit_button("🔄 Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    with col3:
        voice_mode = st.form_submit_button("🎤 Voice", use_container_width=True)
        if voice_mode:
            st.info("Voice input coming soon!")

# Handle user input
if submit_button and user_input.strip():
    # Add user message to history
    user_message = {
        'role': 'user',
        'content': user_input.strip(),
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'id': str(uuid.uuid4())
    }
    st.session_state.chat_history.append(user_message)

    # Show typing indicator
    with st.spinner("🤖 AI is thinking..."):
        time.sleep(1)  # Simulate processing time

        try:
            # Enhanced chat logic
            response_data = process_chat_query(user_input.strip(), dataset_id)

            # Add bot response to history
            bot_message = {
                'role': 'assistant',
                'content': response_data['text_response'],
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'id': str(uuid.uuid4())
            }

            if 'structured_data' in response_data:
                bot_message['structured_data'] = response_data['structured_data']

            st.session_state.chat_history.append(bot_message)

        except Exception as e:
            # Error handling
            error_message = {
                'role': 'assistant',
                'content': f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question.",
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'id': str(uuid.uuid4())
            }
            st.session_state.chat_history.append(error_message)

    # Rerun to update chat display
    st.rerun()

# =====================================================
# ENHANCED CHAT PROCESSING FUNCTION
# =====================================================

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption(
    "🤖 Intelligent ML Assistant • Context-Aware Responses • Data-Driven Insights • "
    f"Dataset: {dataset_id} • Messages: {len(st.session_state.chat_history)}"
)
