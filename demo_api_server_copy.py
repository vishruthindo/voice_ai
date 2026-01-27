"""
Demo FastAPI Server for testing check_prompt22.py with Postman
Run: uvicorn demo_api_server:app --reload --port 8000
Test: POST http://localhost:8000/start-call with JSON body
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import uuid
import re
from bs4 import BeautifulSoup

# Import the LLM generator
from check_prompt22_copy import LLMAnswerGenerator, DocumentRetriever, detect_language

app = FastAPI(title="Voice Bot API Demo")

# -----------------------------
# GLOBAL AGENT CONFIG (UPDATED)
# -----------------------------
agent_config = {
    "bot_name": None,
    "company_name": None,
    "agent_goal": None,
    "default_language": None,
    "support_languages": None,
    "agent_gender": None,
    "guided_conversation": None,
    "tone": "professional",
    "agent_type": "outbound",  # ✅ ADD
}

sessions = {}


# -----------------------------
# REQUEST MODELS
# -----------------------------
class AgentConfigRequest(BaseModel):
    bot_name: Optional[str] = None
    company_name: Optional[str] = None
    agent_goal: Optional[str] = None
    default_language: Optional[str] = None
    support_languages: Optional[List[str]] = None
    agent_gender: Optional[str] = None
    guided_conversation: Optional[str] = None
    tone: Optional[str] = "professional"
    agent_type: Optional[str] = None  # ✅ ADD


class StartCallRequest(BaseModel):
    customer_details: Optional[dict] = None


class MessageRequest(BaseModel):
    session_id: str
    message: str


# -----------------------------
# CONFIGURE AGENT
# -----------------------------
@app.post("/configure-agent")
def configure_agent(request: AgentConfigRequest):
    for key, value in request.dict().items():
        if value is not None:
            agent_config[key] = value

    return {
        "status": "success",
        "message": "Agent configured. Now call /start-call with customer data.",
        "config": agent_config,
    }


# -----------------------------
# START CALL
# -----------------------------
@app.post("/start-call")
def start_call(request: StartCallRequest):
    session_id = str(uuid.uuid4())[:8]

    retriever = DocumentRetriever(embedding_files=["embeddings_data.pkl"])
    llm_generator = LLMAnswerGenerator(
        customer_details=request.customer_details,
        bot_name=agent_config.get("bot_name"),
        company_name=agent_config.get("company_name"),
        agent_goal=agent_config.get("agent_goal"),
        default_language=agent_config.get("default_language"),
        support_languages=agent_config.get("support_languages"),
        agent_gender=agent_config.get("agent_gender"),
        guided_conversation=agent_config.get("guided_conversation"),
        tone=agent_config.get("tone", "professional"),
        agent_type=agent_config.get("agent_type"),
    )

    sessions[session_id] = {
        "generator": llm_generator,
        "retriever": retriever,
        "chat_history": [],
        "customer_details": request.customer_details,
        "agent_type": agent_config.get("agent_type", "outbound"),  # ✅ ADD
    }

    return {
        "status": "success",
        "session_id": session_id,
        "agent_type": agent_config.get("agent_type", "outbound"),
    }


# -----------------------------
# MESSAGE
# -----------------------------
@app.post("/message")
def message(request: MessageRequest):
    # Clean the input message
    original_message = request.message
    request.message = re.sub(r"\s+", " ", request.message).strip()
    request.message = re.sub(r"\s+([,.!?;:])", r"\1", request.message)
    if original_message != request.message:
        print(f"Cleaned message: '{original_message}' -> '{request.message}'")

    session = sessions.get(request.session_id)
    if not session:
        return {"status": "error", "message": "Invalid session_id"}

    generator = session["generator"]
    retriever = session["retriever"]
    chat_history = session["chat_history"]
    agent_type = session.get("agent_type")  

    detected_lang = detect_language(request.message)

    relevant_docs = retriever.get_relevant_documents(request.message)

    response_chunks = []
    metadata = None
    for chunk in generator.voice_answer(
        query=request.message,
        documents=relevant_docs,
        conversation_id=request.session_id,
        chat_history=chat_history,
        lang=detected_lang,
    ):
        if isinstance(chunk, str):
            response_chunks.append(chunk)
        elif isinstance(chunk, dict):
            metadata = chunk

    full_response = "".join(response_chunks)

    # Use the full response as plain text
    text_response = full_response.strip()

    # Print each word on its own line in terminal
    for word in text_response.split():
        print(word)

    should_end = generator._decide_end_call(
        chat_history + [(request.message, full_response)],
        request.message,
        full_response,
    )

    chat_history.append(
        {
            "user": request.message,
            "assistant": full_response,
        }
    )

    return {
        "status": "success",
        "session_id": request.session_id,
        "agent_type": agent_type, 
        "user_message": request.message,
        "assistant_response": text_response,
        "call_ended": should_end,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
