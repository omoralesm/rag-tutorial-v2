# rag-tutorial-v2

This project demonstrates a Retrieval-Augmented Generation (RAG) workflow using Python.

## File Summaries

### `query_data.py`
Handles querying the vector database for relevant documents based on user input. It connects to the database, retrieves embeddings for queries, and returns matching results.

### `populate_database.py`
Responsible for populating the vector database with document embeddings. It processes source documents, generates embeddings, and stores them in the database for later retrieval.

### `get_embedding_function.py`
Provides functions to generate embeddings from text using a selected model. It abstracts the embedding process, allowing other modules to easily obtain vector representations of text.

## Usage

1. Use `populate_database.py` to add documents to the database.
2. Query the database with `query_data.py` to retrieve relevant information.

This code is based on these repos:

https://github.com/pixegami/rag-tutorial-v2

https://github.com/mneedham/LearnDataWithMark/tree/main/llamacpp-rag

