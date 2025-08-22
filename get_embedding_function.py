from llama_cpp import Llama
import chromadb.utils.embedding_functions as embedding_functions

# This class is used to create an embedding function for the Llama model.
# It inherits from the EmbeddingFunction class provided by langchain.

class LlamaCppEmbeddingFunction(embedding_functions.EmbeddingFunction):
    # Initialize the embedding function with the model path.
    # The model_path is the path to the Llama model file.
    def __init__(self, model_path: str, embedding: bool = True):
        self.model = Llama(model_path=model_path, embedding=embedding)

    # This method is used to embed a list of documents.
    # It takes a list of texts and returns a list of embeddings.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            # The create_embedding method returns a dictionary, extract the embedding vector
            result = self.model.create_embedding(text)
            embeddings.append(result['data'][0]['embedding'])
        return embeddings
    
    # This method is used to embed a single query.
    # It takes a query string and returns an embedding vector.
    def embed_query(self, query: str) -> list[float]:
        result = self.model.create_embedding(query)
        return result['data'][0]['embedding']
        