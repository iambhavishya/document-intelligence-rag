import os
import shutil
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class RAGBackend:
    def __init__(self, file_path: str):
        self.file_path = file_path
        
        # Unique ID prevents "tenant" and "database locked" errors
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
        
        # Increased chunk size to avoid API Quota (429) errors
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = text_splitter.split_documents(documents)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✅ Success: Indexed into {self.persist_directory}")

    def get_response(self, query: str, chat_history: list):
        """Processes query with conversation history and returns (answer, sources)."""
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # 1. CONTEXTUALIZE QUESTION: 
        # Reformulates the user's question to be standalone based on chat history.
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # 2. ANSWER QUESTION:
        # Generates the final answer using retrieved context and history.
        system_prompt = (
            "You are an expert assistant. Use the following pieces of retrieved "
            "context to answer the question. If you don't know the answer, say so."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # Final Retrieval Chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 3. EXECUTE
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        
        # 4. EXTRACT SOURCES (Page Numbers)
        sources = []
        for doc in result["context"]:
            # PDF pages are 0-indexed in metadata, so we add 1
            page_num = doc.metadata.get("page", 0) + 1
            sources.append(f"Page {page_num}")
            
        return result["answer"], list(set(sources))
