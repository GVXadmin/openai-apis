from openai import OpenAI, AsyncOpenAI
import re
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional, Dict, AsyncGenerator
from fastapi.middleware.cors import CORSMiddleware
import asyncio

class Settings(BaseSettings):
    OPENAI_API_KEY: str = " "
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
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
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
    assistant_id: Optional[str] = settings.ASSISTANT_ID_PROVIDER
    thread: Dict[str, str]

@app.post("/ask_question")
async def ask_question(request: AskQuestionRequest, auth: bool = Depends(token_authentication)):
    question = request.question
    thread = request.thread
    assistant_id = request.assistant_id

    client.beta.threads.messages.create(
        thread_id=thread["id"],
        role="user",
        content=question
    )

    run = client.beta.threads.runs.create(
        thread_id=thread["id"],
        assistant_id=assistant_id
    )

    while run.status != "completed":
        await asyncio.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread["id"],
            run_id=run.id
        )

    messages = client.beta.threads.messages.list(
        thread_id=thread["id"],
        order="desc"
    )

    resp = messages.data[0].content[0].text.value
    resp = re.sub(r'【.*?】', '', resp)
    return {"response": resp}

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
                    yield f"{event}\n\n"

        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(event_stream(), media_type="text/event-stream")