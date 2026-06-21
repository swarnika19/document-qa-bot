import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        result = genai.embed_content(
            model=self.model_name,
            content=list(input),
        )
        return result["embedding"]