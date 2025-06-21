from pydantic import BaseModel


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    content: str