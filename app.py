import streamlit as st
import os
from backend_logic import RAGBackend

# 1. Page Configuration
st.set_page_config(page_title="Gemini Doc-Intel", page_icon="📄", layout="wide")

st.title("📄 Document Intelligence RAG")
st.markdown("---")

# 2. Sidebar for File Upload & Processing
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        # Save file temporarily to disk for PyPDFLoader
        temp_path = "temp_upload.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Index Document"):
            with st.spinner("Analyzing and Vectorizing..."):
                # Reset chat history for a new document
                st.session_state.messages = []
                # Initialize Backend
                st.session_state.rag = RAGBackend(temp_path)
                st.session_state.rag.process_document()
                st.success("Document Ready!")

# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.caption(f"📍 Sources: {', '.join(message['sources'])}")

# 5. Chat Input Logic
if prompt := st.chat_input("Ask about the document..."):
    if "rag" not in st.session_state:
        st.error("Please upload and index a document first!")
    else:
        # Add User Message to UI State
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI Response
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                # Pass message history to backend for conversational memory
                history_buffer = []
                
                # FIX: Loop through all messages EXCEPT the last one (the current prompt)
                # This prevents sending the same question as both history and input.
                for m in st.session_state.messages[:-1]:
                    if m["role"] == "user":
                        history_buffer.append(("human", m["content"]))
                    else:
                        history_buffer.append(("ai", m["content"]))
                
                # Get answer and sources from backend
                answer, sources = st.session_state.rag.get_response(prompt, history_buffer)
                
                # Display Answer
                st.markdown(answer)
                # Display Sources
                if sources:
                    st.caption(f"📍 Sources: {', '.join(sources)}")
        
        # Save Assistant Message to State
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "sources": sources
        })
