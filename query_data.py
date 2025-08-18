from langchain.vectorstores.chroma import Chroma
import llama_cpp

from get_embedding_function import LlamaCppEmbeddingFunction

CHROMA_PATH = "D:/LLM/Chroma"
MODEL_EMBEDDING_PATH = "D:/LLM/Models/all-MiniLM-L6-v2.F16.gguf"
MODEL_GPT_PATH = "D:/LLM/Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("search_query", type=str, help="The query text.")
    # args = parser.parse_args()
    # search_query = args.search_query
    # query_rag("Which market serves Fabasoft?")
    # query_rag("Since when is the Forrester ECM survey done?")
    # query_rag("Which vendors are included in this assesment?")
    query_rag("Which are the strong performers in the assesment?")


def query_rag(search_query: str):
    # Prepare the DB.
    embedding_function = LlamaCppEmbeddingFunction(model_path=MODEL_EMBEDDING_PATH)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    query_vector = db.similarity_search_with_score(search_query, k=5)
    
    # Create the model.
    model =  llama_cpp.Llama(model_path=MODEL_GPT_PATH, verbose=False, n_ctx=2048)

    # Query the model with the context and the question.
    # Use the stream parameter to get a streaming response.
    # Note: The model's create_chat_completion method is used for chat-based models.
    stream = model.create_chat_completion(
        messages = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(
            context = "\n\n---\n\n".join([doc.page_content for doc, _score in query_vector]),
            question = search_query      
        )}
        ],
        stream=True
    )
    
    response = ""

    for chunk in stream:
        response += chunk['choices'][0]['delta'].get('content', '')
    
    # sources = [doc.metadata.get("id", None) for doc, _score in query_vector]
    sources = ""
    for doc, _score in query_vector:
        sources += "Id: " + doc.metadata.get('id', None) + "\nContent: " + doc.page_content + "\nScore: " + str(_score) + "\n"
    formatted_response = f"Response: {response}\nSources\n {sources}"
    print(formatted_response)
    return response

if __name__ == "__main__":
    main()
