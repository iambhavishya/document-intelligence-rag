
# 📄 Document Intelligence RAG

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://document-intelligence-rag-anyc5cpkunzhzctycvf4nf.streamlit.app/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LLM: Gemini 3](https://img.shields.io/badge/LLM-Gemini%202.5%20/%203-orange.svg)](https://deepmind.google/technologies/gemini/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A professional-grade Retrieval-Augmented Generation (RAG) application designed for high-accuracy document analysis. Built on the **2026 Gemini Flash stack**, this tool transforms static PDFs into interactive, conversational knowledge bases with verifiable source tracking.

### 🔗 [Live Demo: Experience the App Here](https://document-intelligence-rag-anyc5cpkunzhzctycvf4nf.streamlit.app/)

---

## 🚀 Key Features

* **🧠 Conversational Reasoning:** Uses a "Contextualizer" engine to rephrase follow-up questions based on chat history, allowing for fluid, natural multi-turn dialogue.
* **📍 Verifiable Citations:** Every answer includes direct page-number references, ensuring all AI responses are grounded in the source text and minimizing hallucinations.
* **🛡️ Production-Grade Resilience:** Custom-engineered logic to gracefully handle **Google API Free Tier rate limits (429)** and **Internal Server hiccups (500)** via intelligent request pacing, batch processing, and auto-retry fallbacks.
* **🔒 Session Isolation:** Implements unique UUID-based ChromaDB vector stores for every user session, ensuring zero data leakage between different users and document uploads.
* **⚡ Blazing Fast UI:** Built entirely in Python using Streamlit, offering a clean, responsive chat interface with minimal latency.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **LLM** | Google Gemini Flash | Core reasoning and response generation |
| **Embeddings** | Gemini-Embedding-001 | Converting text chunks into vector representations |
| **Orchestration** | LangChain (LCEL) | Chaining prompts, retrievers, and LLM calls |
| **Vector Database** | ChromaDB | Local, ephemeral vector storage for fast semantic search |
| **UI Framework** | Streamlit | Front-end web application interface |

---

## ⚙️ Installation & Local Setup

Want to run this locally? Follow these steps:

### 1. Clone the repository
```bash
git clone [https://github.com/iambhavishya/document-intelligence-rag.git](https://github.com/iambhavishya/document-intelligence-rag.git)
cd document-intelligence-rag

```

### 2. Set up a virtual environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here

```

### 5. Launch the application

```bash
streamlit run app.py

```

The app will automatically open in your default browser at `http://localhost:8501`.

---

## 💡 How to Use

1. **Upload:** Drag and drop your PDF document(s) into the sidebar.
2. **Process:** Wait a few seconds for the app to chunk the text, generate embeddings, and build the temporary vector database.
3. **Query:** Start asking questions! You can ask for summaries, specific data points, or conceptual explanations based on the uploaded file.
4. **Verify:** Check the citations provided at the end of the AI's responses to verify the information on the specific PDF pages.

---

## 🧩 Architecture Overview

1. **Document Loading:** PyPDFLoader extracts text and metadata (like page numbers) from uploaded files.
2. **Chunking:** RecursiveCharacterTextSplitter breaks the document into optimal, semantically complete chunks.
3. **Embedding & Storage:** Chunks are vectorized using Google's embedding model and stored in an isolated ChromaDB collection.
4. **Retrieval:** User queries are embedded and matched against the vector database using similarity search.
5. **Generation:** The context (retrieved chunks) and chat history are passed to Gemini Flash to generate an accurate, grounded, and conversational response.

---

## 👨‍💻 Author

**Bhavishya Grover**

* GitHub: [@iambhavishya](https://www.google.com/search?q=https://github.com/iambhavishya&utm_source=gemini)

*Feel free to open an issue or submit a pull request if you have suggestions for improvements!*

```

```
