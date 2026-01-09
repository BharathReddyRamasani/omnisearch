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
def process_chat_query(question, dataset_id):
    """
    Enhanced chat processing with multiple data sources and response types
    """
    question_lower = question.lower()

    # Get available data
    try:
        # Fetch model metadata
        meta_resp = requests.get(f"{API}/meta/{dataset_id}", timeout=5)
        model_meta = meta_resp.json() if meta_resp.status_code == 200 else {}

        # Fetch EDA data
        eda_resp = requests.get(f"{API}/eda/{dataset_id}", timeout=5)
        eda_data = eda_resp.json() if eda_resp.status_code == 200 else {}

        # Fetch dataset info
        info_resp = requests.get(f"{API}/datasets/{dataset_id}/info", timeout=5)
        dataset_info = info_resp.json() if info_resp.status_code == 200 else {}

    except Exception as e:
        return {
            'text_response': f"I couldn't access the data right now. Error: {str(e)}",
            'structured_data': None
        }

    # Process different types of questions
    response_data = {'text_response': '', 'structured_data': None}

    # Model performance questions
    if any(word in question_lower for word in ['accuracy', 'score', 'performance', 'how good', 'evaluate']):
        if model_meta:
            best_score = model_meta.get('best_score', 0)
            best_model = model_meta.get('best_model', 'Unknown')
            task = model_meta.get('task', 'Unknown')

            response_data['text_response'] = f"Your best model is **{best_model}** with a score of **{best_score:.4f}** for the **{task}** task."

            # Add leaderboard as structured data
            if 'leaderboard' in model_meta:
                response_data['structured_data'] = {
                    'type': 'dataframe',
                    'data': pd.DataFrame(model_meta['leaderboard'])
                }
        else:
            response_data['text_response'] = "You haven't trained any models yet. Go to the Train page to build your ML models!"

    # Feature importance questions
    elif any(word in question_lower for word in ['feature', 'important', 'variable', 'column']):
        if model_meta and 'top_features' in model_meta:
            top_features = model_meta['top_features'][:5]
            response_data['text_response'] = f"The top features for your model are: {', '.join(top_features)}"

            # Add feature importance as structured data if available
            if 'feature_importance' in model_meta:
                features_df = pd.DataFrame(list(model_meta['feature_importance'].items())[:10],
                                         columns=['Feature', 'Importance'])
                response_data['structured_data'] = {
                    'type': 'dataframe',
                    'data': features_df
                }
        else:
            response_data['text_response'] = "Feature importance data isn't available yet. Train a model first!"

    # Data quality questions
    elif any(word in question_lower for word in ['missing', 'null', 'quality', 'clean']):
        if eda_data:
            missing_values = eda_data.get('missing', {})
            total_missing = sum(missing_values.values())
            total_rows = dataset_info.get('rows', 0)

            if total_missing > 0:
                missing_rate = (total_missing / (total_rows * len(missing_values))) * 100
                response_data['text_response'] = f"Your dataset has {total_missing:,} missing values ({missing_rate:.1f}% of total data)."

                # Show missing values breakdown
                missing_df = pd.DataFrame(list(missing_values.items()), columns=['Column', 'Missing Count'])
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                response_data['structured_data'] = {
                    'type': 'dataframe',
                    'data': missing_df
                }
            else:
                response_data['text_response'] = "🎉 Great news! Your dataset has no missing values."
        else:
            response_data['text_response'] = "Run EDA analysis first to check data quality!"

    # Dataset overview questions
    elif any(word in question_lower for word in ['dataset', 'data', 'size', 'shape', 'overview']):
        if dataset_info:
            rows = dataset_info.get('rows', 0)
            cols = dataset_info.get('columns', 0)
            response_data['text_response'] = f"Your dataset has {rows:,} rows and {cols} columns."

            # Add data types as structured data
            if eda_data and 'dtypes' in eda_data:
                dtypes_df = pd.DataFrame(list(eda_data['dtypes'].items()), columns=['Column', 'Data Type'])
                response_data['structured_data'] = {
                    'type': 'dataframe',
                    'data': dtypes_df
                }
        else:
            response_data['text_response'] = "I couldn't access dataset information right now."

    # Model comparison questions
    elif any(word in question_lower for word in ['compare', 'comparison', 'versus', 'vs']):
        if model_meta and 'leaderboard' in model_meta:
            leaderboard = model_meta['leaderboard']
            if len(leaderboard) > 1:
                best = max(leaderboard, key=lambda x: x['score'])
                worst = min(leaderboard, key=lambda x: x['score'])

                response_data['text_response'] = f"Comparing your models: **{best['model']}** performs best ({best['score']:.4f}) while **{worst['model']}** has the lowest score ({worst['score']:.4f})."

                response_data['structured_data'] = {
                    'type': 'dataframe',
                    'data': pd.DataFrame(leaderboard).sort_values('score', ascending=False)
                }
            else:
                response_data['text_response'] = "You only have one trained model. Train more models for comparison!"
        else:
            response_data['text_response'] = "No models to compare. Train some models first!"

    # Prediction questions
    elif any(word in question_lower for word in ['predict', 'forecast', 'estimate']):
        if model_meta:
            response_data['text_response'] = "You can make predictions using your trained model! Go to the Predict page to input new data and get predictions."
        else:
            response_data['text_response'] = "Train a model first before making predictions!"

    # Task type questions
    elif any(word in question_lower for word in ['task', 'type', 'classification', 'regression']):
        if model_meta and 'task' in model_meta:
            task = model_meta['task']
            response_data['text_response'] = f"This is a **{task}** task. "
            if task == 'classification':
                response_data['text_response'] += "You're predicting categories or classes."
            else:
                response_data['text_response'] += "You're predicting continuous numerical values."
        else:
            response_data['text_response'] = "Run model training to determine the task type!"

    # Default response
    else:
        response_data['text_response'] = f"I understand you're asking about: '{question}'. I'm designed to help with questions about your data, model performance, features, and predictions. Try asking something like 'What's the model accuracy?' or 'Show me the top features!'"

    return response_data

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption(
    "🤖 Intelligent ML Assistant • Context-Aware Responses • Data-Driven Insights • "
    f"Dataset: {dataset_id} • Messages: {len(st.session_state.chat_history)}"
)
