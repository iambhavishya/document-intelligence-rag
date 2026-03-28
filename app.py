langchain_google_genai._common.GoogleGenerativeAIError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/document-intelligence-rag/app.py", line 67, in <module>
    answer, sources = st.session_state.rag.get_response(prompt, history_buffer)
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
File "/mount/src/document-intelligence-rag/backend_logic.py", line 127, in get_response
    raise e
File "/mount/src/document-intelligence-rag/backend_logic.py", line 102, in get_response
    retrieved_docs = retriever.invoke(standalone_question)
File "/home/adminuser/venv/lib/python3.14/site-packages/langchain_core/retrievers.py", line 222, in invoke
    result = self._get_relevant_documents(
        input, run_manager=run_manager, **kwargs_
    )
File "/home/adminuser/venv/lib/python3.14/site-packages/langchain_core/vectorstores/base.py", line 1045, in _get_relevant_documents
    docs = self.vectorstore.similarity_search(query, **kwargs_)
File "/home/adminuser/venv/lib/python3.14/site-packages/langchain_chroma/vectorstores.py", line 748, in similarity_search
    docs_and_scores = self.similarity_search_with_score(
        query,
    ...<2 lines>...
        **kwargs,
    )
File "/home/adminuser/venv/lib/python3.14/site-packages/langchain_chroma/vectorstores.py", line 848, in similarity_search_with_score
    query_embedding = self._embedding_function.embed_query(query)
File "/home/adminuser/venv/lib/python3.14/site-packages/langchain_google_genai/embeddings.py", line 490, in embed_query
    raise GoogleGenerativeAIError(msg) from e
