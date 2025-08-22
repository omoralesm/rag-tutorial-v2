import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import LlamaCppEmbeddingFunction
from chromadb.utils.embedding_functions import Chroma

# This script populates a Chroma database with PDF documents.
# It can also clear the database if the --reset flag is provided.
# This path is where the Chroma database will be stored.
# It should be a directory that exists on your system.
CHROMA_PATH = "D:/LLM/Chroma"
# This path is where the PDF documents are stored.
DATA_PATH = "D:/LLM/Data"
# This path is where the embedding model is stored.
MODEL_PATH = "D:/LLM/Models/all-MiniLM-L6-v2.F16.gguf"

# This is the main function that runs when the script is executed.
# It checks if the database should be cleared and then populates it with documents.
def main():

    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

# This function loads PDF documents from the specified directory.
# It uses the PyPDFDirectoryLoader to load all PDF files in the DATA_PATH directory.
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# This function splits the loaded documents into smaller chunks.
# It uses the RecursiveCharacterTextSplitter to split each document into chunks of 800 characters with an overlap of 80 characters.
# This is useful for creating smaller, manageable pieces of text for embedding.
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# This function adds the document chunks to the Chroma database.
# It initializes the LlamaCppEmbeddingFunction with the model path,
def add_to_chroma(chunks: list[Document]):
     # Initialize custom embedding function
    llama_ef = LlamaCppEmbeddingFunction(model_path=MODEL_PATH)
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=llama_ef
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # If there are new chunks, add them to the database.
    # If there are no new chunks, print a message indicating that no new documents were added
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/[PDF file name].pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

# This function clears the Chroma database by deleting the directory where it is stored.
# It uses shutil.rmtree to remove the directory and all its contents.
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
