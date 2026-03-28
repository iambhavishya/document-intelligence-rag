import os
import shutil
import uuid
import time  # <--- MUST IMPORT TIME
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGBackend:
    def __init__(self, file_path: str):
        self.file_path = file_path
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        
        unique_id = str(uuid.uuid4())[:8]
        self.persist_directory = f"./chroma_db_{unique_id}"
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            task_type="retrieval_document" 
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        self.vector_store = None

    def process_document(self):
        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)

            loader = PyPDFLoader(self.file_path)
            documents = loader.load()
            
            # 1. Filter out empty pages that crash the embedding model
            documents = [doc for doc in documents if doc.page_content.strip()]
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)

            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

            # 2. Rate Limit Protection (Avoid 429 Errors)
            batch_size = 15 # Larger batch = fewer total requests
            
            # Optional: Add a UI progress text so you know it's not frozen
            status = st.empty()
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                status.text(f"Indexing batch {(i//batch_size)+1} ... (Pacing API)")
                
                self.vector_store.add_documents(batch)
                
                # The Magic Pause: Wait 2.5 seconds between requests
                time.sleep(2.5) 
                
            status.text("✅ Successfully indexed!")

        except Exception as e:
            # 3. UNMASK THE ERROR: This forces the real error onto your screen
            st.error(f"🚨 GOOGLE API ERROR DETAILS: {str(e)}")
            raise e

    # ... (Keep your get_response method exactly the same) ...
