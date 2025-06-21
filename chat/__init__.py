from chat.api import ChatController
from chat.service import RAGService


def chat_router(deps: dict):
    openai_client = deps["openai_client"]
    # groq_client= deps["groq_client"]
    cohere_client = deps["cohere_client"]
    qdrant_client = deps["qdrant_client"]
    embedder = deps["embedder"]

    rag_svc = RAGService(
        # llm_client=openai_client,
        llm_client=openai_client,
        vectordb_client=qdrant_client,
        reranker_client=cohere_client,
        embedder=embedder,
    )
    controller = ChatController(rag_svc=rag_svc)

    return controller.router