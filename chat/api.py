from fastapi import APIRouter

from llmsmith.task.models import TaskOutput

from chat.model import ChatRequest, ChatResponse
from chat.service import RAGService


class ChatController:
    def __init__(self, rag_svc: RAGService) -> None:
        self.rag_svc = rag_svc
        self.router: APIRouter = APIRouter(tags=["Chat endpoint"], prefix="/api")

        self.router.add_api_route(
            path="/chat",
            endpoint=self.chat,
            methods=["POST"],
        )

    async def chat(self, req_body: ChatRequest):
        rag_response: TaskOutput = await self.rag_svc.chat(req_body.content)
        return ChatResponse(content=rag_response.content)