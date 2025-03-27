import os
from openai import OpenAI, AsyncOpenAI
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional, Dict, AsyncGenerator
from fastapi.middleware.cors import CORSMiddleware
from app.services.schedule_appointment import handle_appointment_workflow
from app.services.ask_question import process_question
from app.services.detect_intent import detect_intent, detect_source
import json

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    VECTOR_STORE_ID: str = os.getenv("VECTOR_STORE_ID")
    VECTOR_STORE_ID_PATIENT: str = os.getenv("VECTOR_STORE_ID_PATIENT")
    ASSISTANT_ID_PATIENT: str = os.getenv("ASSISTANT_ID_PATIENT")
    ASSISTANT_ID_PROVIDER: str = os.getenv("ASSISTANT_ID_PROVIDER")

settings = Settings()
app = FastAPI()
client = OpenAI(api_key=settings.OPENAI_API_KEY)
async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Custom exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors(), "body": exc.body},
    )

# Token authentication
def token_authentication(authorization: Optional[str] = Header(None)):
    if authorization is None or authorization != "Bearer U6P9tG5m8iY387Z9QN7LAFld":
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return True

class SetupAssistantRequest(BaseModel):
    assistant_id: Optional[str] = settings.ASSISTANT_ID_PROVIDER

@app.post("/setup_assistant")
async def setup_assistant(request: SetupAssistantRequest, auth: bool = Depends(token_authentication)):
    try:
        assistant_id = request.assistant_id
        thread = client.beta.threads.create()
        return {
            "client_info": str(client),
            "vector_store_id": settings.VECTOR_STORE_ID,
            "assistant_id": assistant_id,
            "thread": thread.id if hasattr(thread, 'id') else str(thread)
        }
    except Exception as e:
        return {"error": str(e)}

class AskQuestionRequest(BaseModel):
    question: str
    assistant_id: Optional[str]
    is_clinician: Optional[bool] = False
    thread: Dict[str, str]

conversation_store = {}  # Tracks conversation history
workflow_store = {}  # Tracks active workflow per thread
source_store = {}

QUESTION_PHRASES = {"ask a question about health or obesity", "ask a question", "i have a question", "i need help", "i need assistance", "can you assist me?", "i need some info"}

@app.post("/ask_question")
async def ask_question(request: AskQuestionRequest, auth: bool = Depends(lambda: True)):
    thread_id = request.thread["id"]

    # Initialize session tracking if not exists
    if thread_id not in conversation_store:
        conversation_store[thread_id] = []
    if thread_id not in workflow_store:
        workflow_store[thread_id] = None  # No active workflow at start

    user_input = request.question.strip()

    if user_input in QUESTION_PHRASES:
        workflow_store[thread_id] = "general_question"
        return json.dumps({
            "Message": "Sure! Go ahead, how can I assist you today?",
            "input_type": "text"
        })

    if workflow_store[thread_id] == "appointment":
        response_data = handle_appointment_workflow(user_input)
        
        if response_data.get("is_api_call", False):  
            workflow_store[thread_id] = None  # Reset flow after appointment is finalized
        
        return json.dumps(response_data) 
    
    if workflow_store[thread_id] == "general_question":
        intent = await detect_intent(user_input)  
        # If user suddenly wants to book an appointment mid-conversation, the assistant should switch to the appointment workflow
        if intent == "appointment_booking":
            workflow_store[thread_id] = "appointment"  
            return json.dumps(handle_appointment_workflow("Book an Appointment"))  
        # If user suddenly switches to small talk, salutation or greeting
        if intent == "unclear":
            return json.dumps({
                "Message": "Hello! How may I assist you today?",
                "input_type": "options",
                "Options": [
                    {"Id": 1, "Option": "Schedule an Appointment"},
                    {"Id": 2, "Option": "Ask a question about health or obesity"}
                ]
            })
        
        if thread_id not in source_store:
            source_store[thread_id] = "general"

        if source_store[thread_id] == "general":
            source_store[thread_id] = await detect_source(user_input)

        source = source_store[thread_id]
        # print("Source selected:", source)

        response_data = await process_question(user_input, request.is_clinician, conversation_store[thread_id], source)
        return json.dumps(response_data)   

    # Detect intent only if no active workflow
    intent = await detect_intent(user_input)

    if intent == "appointment_booking":
        workflow_store[thread_id] = "appointment" 
        return json.dumps(handle_appointment_workflow("Book an Appointment"))  
    
    elif intent == "general_question":
        workflow_store[thread_id] = "general_question"  
        source = await detect_source(user_input)
        response_data = await process_question(user_input, request.is_clinician, conversation_store[thread_id], source)
        return json.dumps(response_data)  

    # Note for later: introduce multi-intent check, if a user gives a response that can be interpreted as both appointment booking and general question
    # Default: Show greeting options again if intent is unclear
    response_data = {
        "Message": "Hello! What would you like to do today?",
        "input_type": "options",
        "Options": [
            {"Id": 1, "Option": "Schedule an Appointment"},
            {"Id": 2, "Option": "Ask a question about health or obesity"}
        ]
    }
    return json.dumps(response_data)  

@app.post("/get_stream_response")
async def assistant_stream(request: AskQuestionRequest, auth: bool = Depends(token_authentication)):
    async def event_stream() -> AsyncGenerator[str, None]:
        question = request.question
        thread = request.thread
        assistant_id = request.assistant_id

        try:
            await async_client.beta.threads.messages.create(
                thread_id=thread['id'],
                role="user",
                content=question
            )

            stream = async_client.beta.threads.runs.stream(
                thread_id=thread['id'],
                assistant_id=assistant_id
            )

            async with stream as response_stream:
                async for event in response_stream.text_deltas:
                    yield f"data: {event}\n\n"
                    yield f"{event}\n\n"

        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
