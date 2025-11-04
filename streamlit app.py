import streamlit as st
import requests
import uuid

# -------------------------
# CONFIG
# -------------------------
CHAT_API_URL = "http://localhost:8000/chat"   # backend endpoint

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# -------------------------
# SESSION STATE INIT
# -------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of dicts: {"role": "...", "text": "..."}

# -------------------------
# SIDEBAR INFO
# -------------------------
with st.sidebar:
    st.title("🔍 RAG Chatbot")
    st.markdown("This Streamlit app interacts with your LangChain RAG backend.")
    st.write("**Session ID:**")
    st.code(st.session_state.session_id)

    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# -------------------------
# TITLE
# -------------------------
st.markdown("<h1 style='text-align:center;'>🤖 Intelligent RAG Chat Assistant</h1>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# CHAT HISTORY DISPLAY
# -------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["text"])
    else:
        st.chat_message("assistant").markdown(msg["text"])

# -------------------------
# CHAT INPUT
# -------------------------
user_query = st.chat_input("Ask anything from your knowledge base...")

if user_query:
    # Show message instantly
    st.session_state.messages.append({"role": "user", "text": user_query})
    st.chat_message("user").markdown(user_query)

    # -----------------------------
    # CALL BACKEND CHAT API
    # -----------------------------
    try:
        payload = {
            "session_id": st.session_state.session_id,
            "question": user_query
        }
        response = requests.post(CHAT_API_URL, json=payload)
        data = response.json()

        answer = data["answer"]
        st.session_state.session_id = data["session_id"]

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "text": answer})

        # Render assistant message
        st.chat_message("assistant").markdown(answer)

    except Exception as e:
        st.error(f"Error calling Chat API: {e}")