# webapp/backend/services/chatbot_service.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from utils.chatbot import chatbot_response
import uuid

def process_message(message: str, session_id: str = None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    formatted_response, empathetic_reply = chatbot_response(message, session_id)
    return formatted_response, empathetic_reply


