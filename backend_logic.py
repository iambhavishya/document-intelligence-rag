import os
import shutil
import uuid
import time
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
        
        # 1. Set the API key globally to avoid validation errors
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

        # --- 🚨 NEW DEPENDENCY-FREE DEBUG BLOCK 🚨 ---
        import requests
        api_key = st.secrets["GOOGLE_API_KEY"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(url).json()
        
        if 'models' in response:
            valid_models = [m['name'] for m in response['models'] if 'generateContent' in m.get('supportedGenerationMethods', [])]
            st.error(f"✅ YOUR VALID CHAT MODELS ARE: {valid_models}")
        else:
            st.error(f"🚨 API ERROR: {response}")
        st.stop() # Halts the app so you can read the list
        # ---------------------------------------------
        
        unique_id = str(uuid.uuid4())[:8]
        # ... (rest of your init code stays the same)
        self.persist_directory = f"./chroma_db_{unique_id}"
        
        # 2. Use the standard 2026 text-embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            task_type="retrieval_document" 
        )
        
        self.llm = ChatGoogleGenerativeAI(model="gemini-3-flash", temperature=0.3)
        self.vector_store = None

    def process_document(self):
        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)

            loader = PyPDFLoader(self.file_path)
            documents = loader.load()
            
            # Filter out empty pages that crash the embedding API
            documents = [doc for doc in documents if doc.page_content.strip()]
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)

            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

            # Process in small batches to respect Google's Free Tier Rate Limits
            batch_size = 15
            status = st.empty()
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                status.text(f"Indexing batch {(i//batch_size)+1} of {(len(chunks)//batch_size)+1} ...")
                
                self.vector_store.add_documents(batch)
                
                # Pace the API to avoid 429 Quota errors
                time.sleep(2.5) 
                
            status.text("✅ Successfully indexed!")

        except Exception as e:
            st.error(f"🚨 GOOGLE API ERROR DETAILS: {str(e)}")
            raise e

    def get_response(self, query: str, chat_history: list):
        """Processes the user query and retrieves answers from the PDF."""
        try:
            if not self.vector_store:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory, 
                    embedding_function=self.embeddings
                )

            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

            # --- Contextualization Step ---
            context_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given chat history and a question, rephrase it as a standalone question. Do NOT answer it."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            context_chain = context_prompt | self.llm | StrOutputParser()
            standalone_question = context_chain.invoke({"input": query, "chat_history": chat_history})

            # --- Retrieval Step ---
            retrieved_docs = retriever.invoke(standalone_question)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # --- Answer Generation Step ---
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant. Answer the question using ONLY the provided context:\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            qa_chain = qa_prompt | self.llm | StrOutputParser()
            
            answer = qa_chain.invoke({
                "input": standalone_question, 
                "chat_history": chat_history, 
                "context": context_text
            })

            # --- Citation Extraction ---
            sources = [f"Page {doc.metadata.get('page', 0) + 1}" for doc in retrieved_docs]
            
            return answer, list(set(sources))

        except Exception as e:
            # THIS WILL PRINT THE EXACT REASON GOOGLE REJECTED THE CHAT
            st.error(f"🚨 GOOGLE CHAT API ERROR DETAILS: {str(e)}")
            raise e
