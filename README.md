# 📄 Document Intelligence RAG
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://app-learning-mipnqfqnhaenphtex7xppp.streamlit.app/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LLM: Gemini 3](https://img.shields.io/badge/LLM-Gemini%202.5%20/%203-orange.svg)](https://deepmind.google/technologies/gemini/)

A professional-grade Retrieval-Augmented Generation (RAG) application designed for high-accuracy document analysis. Built on the **2026 Gemini Flash stack**, this tool transforms static PDFs into interactive, conversational knowledge bases with verifiable source tracking.

### 🔗 [Live Demo: Experience the App Here](https://app-learning-mipnqfqnhaenphtex7xppp.streamlit.app/)

---

## 🚀 Key Features

* **🧠 Conversational Reasoning:** Uses a "Contextualizer" engine to rephrase follow-up questions based on chat history, allowing for fluid, natural dialogue.
* **📍 Verifiable Citations:** Every answer includes direct page-number references, ensuring all AI responses are grounded in the source text.
* **🛡️ Production-Grade Resilience:** Custom-engineered logic to handle **Google API Free Tier rate limits (429)** and **Internal Server hiccups (500)** via intelligent request pacing, batch processing, and auto-retry fallbacks.
* **🔒 Session Isolation:** Implements unique UUID-based ChromaDB vector stores for every user session, ensuring zero data leakage between different document uploads.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM** | Google Gemini 2.5 / 3 Flash |
| **Embeddings** | Gemini-Embedding-001 |
| **Orchestration** | LangChain (LCEL) |
| **Vector Database** | ChromaDB |
| **UI Framework** | Streamlit |

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/iambhavishya/document-intelligence-rag.git](https://github.com/iambhavishya/document-intelligence-rag.git)
cd document-intelligence-rag
