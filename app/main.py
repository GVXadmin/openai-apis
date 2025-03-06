from openai import OpenAI, AsyncOpenAI
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional, Dict, AsyncGenerator
from fastapi.middleware.cors import CORSMiddleware
from services.schedule_appointment import handle_appointment_workflow
from services.ask_question import process_question
from services.detect_intent import detect_intent
import json

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    VECTOR_STORE_ID: str = ""
    VECTOR_STORE_ID_PATIENT: str = ""
    ASSISTANT_ID_PATIENT: str = ""
    ASSISTANT_ID_PROVIDER: str = ""

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

# Store conversation history per thread (session tracking)

conversation_store = {}  # Tracks conversation history
workflow_store = {}  # Tracks active workflow per thread

@app.post("/ask_question")
async def ask_question(request: AskQuestionRequest, auth: bool = Depends(lambda: True)):
    thread_id = request.thread["id"]

    # Initialize session tracking if not exists
    if thread_id not in conversation_store:
        conversation_store[thread_id] = []
    if thread_id not in workflow_store:
        workflow_store[thread_id] = None  # No active workflow at start

    user_input = request.question.strip().lower()

    if workflow_store[thread_id] == "appointment":
        response_data = handle_appointment_workflow(user_input)
        
        if response_data.get("is_api_call", False):  
            workflow_store[thread_id] = None  
        
        return json.dumps(response_data) 
    
    if workflow_store[thread_id] == "general_question":
        intent = await detect_intent(user_input)  

        if intent == "appointment_booking":
            workflow_store[thread_id] = "appointment"  
            return json.dumps(handle_appointment_workflow("Book an Appointment"))  

        response_data = await process_question(user_input, request.is_clinician, conversation_store[thread_id])
        return json.dumps(response_data)  

    # Detect intent only if no active workflow
    intent = await detect_intent(user_input)

    if intent == "appointment_booking":
        workflow_store[thread_id] = "appointment" 
        return json.dumps(handle_appointment_workflow("Book an Appointment"))  
    
    elif intent == "general_question":
        workflow_store[thread_id] = "general_question"  
        response_data = await process_question(user_input, request.is_clinician, conversation_store[thread_id])
        return json.dumps(response_data)  

    # Default: Show greeting options again if unclear
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