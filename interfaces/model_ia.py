from ai_algorithm.chatbot_model import ChatbotModel

class ChatbotInterface:
  def __init__(self):
    self.model = ChatbotModel()

  def get_chatbot_response(self, message: str):
    greeting = self.model.check_for_greeting(message)
    if greeting:
      return greeting
    return self.model.get_response(message)
