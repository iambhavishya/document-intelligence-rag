import os
import shutil
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.run_nables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGBackend:
    def __init__(self, file_path: str):
        self.file_path = file_path
        
        # Unique ID prevents "database locked" errors in Streamlit
        unique_id = str(uuid.uuid4())[:8]
        self.persist_directory = f"./chroma_db_{unique_id}"
        
        # 2026 Stable Models
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
        self.llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3)
        self.vector_store = None

    def process_document(self):
        """Loads PDF, chunks text, and stores in a fresh ChromaDB."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        
        # Balanced chunk size for performance and API quota
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✅ Indexed into {self.persist_directory}")

    def get_response(self, query: str, chat_history: list):
        """
        Processes query using LCEL (LangChain Expression Language).
        Bypasses 'langchain.chains' to ensure Python 3.14 compatibility.
        """
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # --- 1. CONTEXTUALIZATION STEP ---
        # Turns follow-up questions into standalone questions
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given chat history and a question, rephrase it as a standalone question. Do NOT answer it."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        context_chain = context_prompt | self.llm | StrOutputParser()
        standalone_question = context_chain.invoke({"input": query, "chat_history": chat_history})

        # --- 2. RETRIEVAL STEP ---
        retrieved_docs = retriever.invoke(standalone_question)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # --- 3. ANSWER GENERATION STEP ---
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
