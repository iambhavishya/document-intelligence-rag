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
from langchain_core.output_parsers import StrOutputParser

class RAGBackend:
    def __init__(self, file_path: str):
        self.file_path = file_path
        
        # 1. Set the API key globally to avoid validation errors
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        
        unique_id = str(uuid.uuid4())[:8]
        self.persist_directory = f"./chroma_db_{unique_id}"
        
        # 2. The Current 2026 Embedding Model 
        # (Deliberately leaving out task_type to prevent 500 errors on short chat queries)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        
        # 3. The Correct Chat Model (Direct from your API key's validated list)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.3
        )
        
        self.vector_store = None

    def process_document(self):
        """Loads, chunks, and indexes the PDF into ChromaDB safely."""
        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)

            # Load and clean document
            loader = PyPDFLoader(self.file_path)
            documents = loader.load()
            documents = [doc for doc in documents if doc.page_content.strip()]
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)

            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

            # --- API QUOTA PROTECTION ---
            # Process in larger batches and pause to respect Google's Free Tier Rate Limits
            batch_size = 50
            status = st.empty()
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                status.text(f"Indexing batch {(i//batch_size)+1} of {(len(chunks)//batch_size)+1} ...")
                
                self.vector_store.add_documents(batch)
                
                # Wait 10 seconds between batches to safely pace the Free Tier API
                time.sleep(10) 
                
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

            # --- 1. SMART CONTEXTUALIZATION ---
            # Only ask the LLM to rephrase if there is actually a history to look at.
            if chat_history:
                context_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Given the chat history and the latest question, rephrase it into a short, standalone question. Return ONLY the question, no other text."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                
                context_chain = context_prompt | self.llm | StrOutputParser()
                standalone_question = context_chain.invoke({"input": query, "chat_history": chat_history})
                
                # Clean up any weird formatting the AI might add
                standalone_question = standalone_question.replace("Standalone question:", "").replace("**", "").replace('"', '').strip()
                
                # Fallback if the AI hallucinates a massive string or returns blank
                if not standalone_question or len(standalone_question) > 300:
                    standalone_question = query
                    
                st.caption(f"*(AI Context Engine: {standalone_question})*")
                
                # --- THE SPEED BUMP FIX ---
                # Google's Free Tier throws 500 errors if an LLM call and an Embedding call 
                # happen in the exact same millisecond. Pause to let the server breathe.
                time.sleep(1.5)
                
            else:
                # First question! No history needed to rephrase.
                standalone_question = query

            # --- 2. RETRIEVAL WITH SAFETY FALLBACK ---
            try:
                retrieved_docs = retriever.invoke(standalone_question)
            except Exception as e:
                # If Google STILL 500s on the rephrased query, catch the crash, wait, and retry safely
                if "500" in str(e) or "INTERNAL" in str(e):
                    st.warning("⚠️ Google Server hiccuped on the memory query. Retrying...")
                    time.sleep(1.5)
                    standalone_question = query
                    retrieved_docs = retriever.invoke(standalone_question)
                else:
                    raise e

            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # --- 3. ANSWER GENERATION ---
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

            # --- 4. CITATION EXTRACTION ---
            sources = [f"Page {doc.metadata.get('page', 0) + 1}" for doc in retrieved_docs]
            
            return answer, list(set(sources))

        except Exception as e:
            st.error(f"🚨 GOOGLE CHAT API ERROR DETAILS: {str(e)}")
            raise e
