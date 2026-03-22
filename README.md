# 📄 Document Intelligence RAG
A professional-grade Retrieval-Augmented Generation (RAG) application built with the **Gemini 3 Flash** stack. This tool allows users to upload PDF documents and engage in context-aware conversations with verifiable source citations.

## 🚀 Key Features
* **Conversational Memory:** Remembers previous questions in a session to handle follow-up queries naturally.
* **Source Citations:** Automatically identifies and displays the specific page numbers used to generate each answer.
* **Session Isolation:** Uses unique UUID-based vector storage to prevent data leakage between different document uploads.
* **Optimized for 2026:** Built using the latest `gemini-3-flash-preview` and `gemini-embedding-2-preview` models.

## 🛠️ Tech Stack
* **LLM:** Google Gemini 3 Flash
* **Orchestration:** LangChain (LCEL)
* **Vector Database:** ChromaDB
* **Frontend:** Streamlit
* **Language:** Python 3.12+

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/iambhavishya/document-intelligence-rag.git](https://github.com/iambhavishya/document-intelligence-rag.git)
   cd document-intelligence-rag