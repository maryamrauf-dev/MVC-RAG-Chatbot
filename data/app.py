import streamlit as st
import os
import rag_core
import uuid
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Calculus RAG Chatbot",
    page_icon="📘",
    layout="wide"
)

# Hide Chat Avatars (CSS)
st.markdown("""
<style>
    [data-testid="stChatMessageAvatar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = {} # {chat_id: {"title": str, "messages": list}}
if "current_chat_id" not in st.session_state:
    # Create the first chat session
    first_id = str(uuid.uuid4())
    st.session_state.history[first_id] = {"title": "New Chat", "messages": []}
    st.session_state.current_chat_id = first_id

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

# Quick access to current messages
current_chat = st.session_state.history[st.session_state.current_chat_id]
messages = current_chat["messages"]

st.title("📘 Calculus AI Tutor")
st.caption("Powered by Google Gemini & RAG")

# Sidebar
with st.sidebar:
    st.header("Knowledge Base")
    if st.button("Reload Knowledge Base"):
        st.cache_resource.clear()
        if os.path.exists(rag_core.DB_FILE):
             os.remove(rag_core.DB_FILE)
        st.rerun()
    st.info("Source: Thomas Finney Calculus (9th Edition)")

    st.divider()
    st.header("💬 Conversations")
    
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.history[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.current_chat_id = new_id
        st.rerun()

    # List history items
    for chat_id, data in reversed(list(st.session_state.history.items())):
        is_current = chat_id == st.session_state.current_chat_id
        
        # Avoid redundancy: don't show the current chat in the list if it's empty
        if is_current and not data["messages"]:
            continue
            
        button_label = data["title"][:25] + "..." if len(data["title"]) > 25 else data["title"]
        if st.button(
            button_label, 
            key=chat_id, 
            use_container_width=True,
            type="secondary"
        ):
            st.session_state.current_chat_id = chat_id
            st.rerun()
    
    st.divider()
    st.header("Attachments")
    uploaded_file = st.file_uploader("Upload a math problem image", type=["png", "jpg", "jpeg"], help="Upload an image of a calculus problem to analyze it with the book.")
    if uploaded_file:
        st.image(uploaded_file, caption="Waiting for your query...", use_container_width=True)

# Load Data
if not st.session_state.db_ready:
    # Check if DB exists
    chunks = rag_core.load_knowledge_base()
    
    if chunks:
        st.session_state.chunks = chunks
        st.session_state.db_ready = True
    else:
        # Build it
        try:
            with st.spinner("Initializing Knowledge Base..."):
                progress_bar = st.progress(0, text="Starting...")
                
                def update_progress(p, text):
                    progress_bar.progress(p, text=text)
                
                st.session_state.chunks = rag_core.build_knowledge_base(update_progress)
                st.session_state.db_ready = True
                progress_bar.empty()
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            st.stop()

# Chat Interface
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"]:
            st.image(message["image"], caption="Uploaded Image", width=400)

if prompt := st.chat_input("Ask a calculus question..."):
    # If this is the first message, update the chat title
    if not messages:
        st.session_state.history[st.session_state.current_chat_id]["title"] = prompt

    # Prepare image
    img_data = None
    if uploaded_file:
        img_data = Image.open(uploaded_file)
    
    # Store user message
    user_msg = {"role": "user", "content": prompt}
    if img_data:
        user_msg["image"] = img_data
        
    messages.append(user_msg)
    
    with st.chat_message("user"):
        st.markdown(prompt)
        if img_data:
            st.image(img_data, caption="Uploaded Image", width=400)

    with st.chat_message("assistant"):
        try:
            # Retrieve multimodal context (page info)
            with st.spinner("Searching textbook..."):
                retrieved_items = rag_core.retrieve_context_multimodal(prompt)
            
            # Generate answer using high-res page images (streaming)
            stream = rag_core.generate_answer(prompt, retrieved_items, img_data)
            full_response = st.write_stream(stream)
            
            messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error: {e}")
