import cohere
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
# from fastembed import TextEmbedding
import openai
import os
import groq
from qdrant_client import AsyncQdrantClient
import uvicorn
from chat import chat_router
from core.log import logger
from config import settings
from langsmith.wrappers import wrap_openai
# from rag_llmsmith_fastapi import app_version



app = FastAPI(version="1.0.0", title="ITB Care RAG App")

os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH.API_KEY
os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH.TRACING

# Create OpenAI client
openai_client = wrap_openai(openai.AsyncOpenAI(api_key=settings.OPENAI.API_KEY))
# Create Groq client
# groq_client = groq.AsyncGroq(api_key=settings.GROQ.API_KEY)
# Create Cohere client
cohere_client = cohere.AsyncClient(api_key=settings.COHERE.API_KEY)
# Create Qdrant client
qdrant_client = AsyncQdrantClient(
    url=settings.QDRANT.URL, api_key=settings.QDRANT.API_KEY
)
# Fastembed is used for embedding the documents inserted into Qdrant.
embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# embedder = TextEmbedding("BAAI/bge-small-en")

dependencies: dict = {
    "openai_client": openai_client,
    # "groq_client": groq_client,
    "cohere_client": cohere_client,
    "qdrant_client": qdrant_client,
    "embedder": embedder,
}

app.include_router(chat_router(dependencies))

logger.info("Hola! I'm ready to receive requests.")

if __name__ == "__main__":
    uvicorn.run("rag_llmsmith_fastapi.main:app", port=8000, log_level="info")