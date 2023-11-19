from fastapi import APIRouter, HTTPException, Request, Body
from interfaces.model_ia import ChatbotInterface
from models.Model import Message

router = APIRouter()

chatbot_interface = ChatbotInterface()

@router.post('/chat')
def chat_with_bot(
  message: Message = Body(...)
):
  if not message:
    raise HTTPException(status_code=400, detail="Message is required")
  response = chatbot_interface.get_chatbot_response(message.message)
  return {"response": response}
