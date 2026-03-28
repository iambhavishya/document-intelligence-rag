import streamlit as st
import os
from backend_logic import RAGBackend
from langchain_core.messages import HumanMessage, AIMessage

# 1. Page Configuration
st.set_page_config(page_title="Gemini Doc-Intel", page_icon="📄", layout="wide")

st.title("📄 Document Intelligence RAG")
st.markdown("---")

# 2. Sidebar for File Upload & Processing
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        temp_path = "temp_upload.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Index Document"):
            with st.spinner("Analyzing and Vectorizing..."):
                st.session_state.messages = []
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
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                history_buffer = []
                
                # Use proper LangChain Message objects so the LLM doesn't hallucinate
                for m in st.session_state.messages[:-1]:
                    if m["role"] == "user":
                        history_buffer.append(HumanMessage(content=m["content"]))
                    else:
                        history_buffer.append(AIMessage(content=m["content"]))
                
                answer, sources = st.session_state.rag.get_response(prompt, history_buffer)
                
                st.markdown(answer)
                if sources:
                    st.caption(f"📍 Sources: {', '.join(sources)}")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "sources": sources
        })
