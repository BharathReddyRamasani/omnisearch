import streamlit as st
import requests
import json
import time
from datetime import datetime
import uuid
import pandas as pd

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="OmniSearch AI – Intelligent Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# MODERN BLUE GRADIENT CSS STYLING
# =====================================================
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
        border: 1px solid rgba(255,255,255,0.1);
    }
    .chat-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .chat-subtitle {
        font-size: 1.1rem;
        margin: 0.75rem 0 0 0;
        opacity: 0.92;
        font-weight: 500;
    }
    .chat-container {
        background: white;
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #e5e7eb;
    }
    .message-user {
        background: linear-gradient(135deg, #0052cc, #2563eb);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.75rem 0;
        margin-left: 15%;
        box-shadow: 0 4px 12px rgba(5, 82, 204, 0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .message-bot {
        background: #f9fafb;
        color: #1f2937;
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.75rem 0;
        margin-right: 15%;
        border-left: 4px solid #0052cc;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }
    .message-timestamp {
        font-size: 0.75rem;
        opacity: 0.6;
        margin-top: 0.5rem;
    }
    .input-container {
        background: white;
        padding: 1.75rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
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
    OmniSearch AI DSL-based chat processor.
    Returns JSON DSL only - never generates explanatory text.
    """
    try:
        payload = {
            "question": question,
            "history": st.session_state.get('chat_history', [])[-10:]  # Last 10 messages for context
        }
        resp = requests.post(f"{API}/api/chat/{dataset_id}", json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "ok":
                dsl = data.get("dsl", {})
                result = data.get("result")
                
                # Format response based on DSL action
                action = dsl.get("action", "unknown")
                if action == "unsupported":
                    return {
                        'dsl': dsl,
                        'result': None,
                        'error': dsl.get('reason', 'Unsupported query')
                    }
                
                return {
                    'dsl': dsl,
                    'result': result,
                    'error': None
                }
            else:
                return {
                    'dsl': {'action': 'unsupported'},
                    'result': None,
                    'error': data.get('message', 'Unknown error')
                }
        else:
            return {
                'dsl': {'action': 'unsupported'},
                'result': None,
                'error': f"Backend error: {resp.status_code}"
            }
    except Exception as e:
        return {
            'dsl': {'action': 'unsupported'},
            'result': None,
            'error': f"Connection error: {str(e)}"
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
                # Display AI response with DSL info
                if 'dsl' in message:
                    action = message['dsl'].get('action', 'unknown')
                    error = message.get('error')
                    result = message.get('result')
                    
                    # Display the DSL action
                    st.markdown(f"""
                    <div class="message-bot">
                        <strong>AI DSL Query:</strong> {action}
                        <div class="message-timestamp">{message['timestamp']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display error if any
                    if error:
                        st.error(f"⚠️ {error}")
                    
                    # Display DSL parameters as expandable JSON
                    if action != 'unsupported':
                        with st.expander("📋 Query Details (JSON DSL)"):
                            st.json(message['dsl'])
                    
                    # Display result if available
                    if result and not isinstance(result, dict) or (isinstance(result, dict) and 'error' not in result):
                        st.success("✅ Query executed successfully")
                        if isinstance(result, dict):
                            # Format result based on type
                            if 'dataframe' in str(type(result)):
                                st.dataframe(result)
                            else:
                                with st.expander("📊 Results"):
                                    st.json(result, expanded=False)
                        else:
                            st.write(result)
                else:
                    # Legacy message format
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

# Professional input with button
col1, col2 = st.columns([1, 0.15])
with col1:
    user_input = st.text_area(
        "Type your question here...",
        height=100,
        placeholder="Ask me anything about your data, models, or insights...",
        key="chat_input",
        help="Press Enter to submit, or use the send button below"
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    send_button = st.button("🚀", key="send_btn", use_container_width=True, help="Send message")
    clear_button = st.button("🔄", key="clear_btn", use_container_width=True, help="Clear history")

# Handle send button click or user submission
submit_button = send_button

# Handle clear
if clear_button:
    st.session_state.chat_history = []
    st.rerun()


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
        time.sleep(0.5)

        try:
            # Get DSL response from backend
            response_data = process_chat_query(user_input.strip(), dataset_id)

            # Prepare bot message with DSL info
            bot_message = {
                'role': 'assistant',
                'content': f"Query: {response_data['dsl'].get('action', 'unknown')}",
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'id': str(uuid.uuid4()),
                'dsl': response_data['dsl'],
                'result': response_data['result'],
                'error': response_data.get('error')
            }

            st.session_state.chat_history.append(bot_message)

        except Exception as e:
            # Error handling
            error_message = {
                'role': 'assistant',
                'content': f"Error processing query: {str(e)}",
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'id': str(uuid.uuid4()),
                'error': str(e)
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
