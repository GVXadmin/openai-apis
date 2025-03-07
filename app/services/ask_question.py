import asyncio
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

AZURE_OPENAI_TEXTEMBEDDER_ENDPOINT = os.getenv("AZURE_OPENAI_TEXTEMBEDDER_ENDPOINT")
AZURE_OPENAI_TEXTEMBEDDER_API_VERSION = os.getenv("AZURE_OPENAI_TEXTEMBEDDER_API_VERSION")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment="text-embedding-3-large",
    openai_api_version=AZURE_OPENAI_TEXTEMBEDDER_API_VERSION, 
    azure_endpoint=AZURE_OPENAI_TEXTEMBEDDER_ENDPOINT  
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="fusioncare-rag",
    url=QDRANT_URL,
    prefer_grpc=True,
    api_key=QDRANT_API_KEY
)

async def process_question(question: str, is_clinician: bool, conversation_history: list) -> str:
  
    matches = await asyncio.to_thread(vector_store.similarity_search, question, 5)
    context_chunks = [match.page_content for match in matches]

    if not context_chunks:  
        return "I do not have the necessary context in my knowledge base. Please contact customer support for further assistance."

    context_text = "\n\n".join(context_chunks)

    system_messages = [{
        "role": "system",
        "content": (
            "You are an AI assistant designed to help patients by providing accurate and reliable answers. "
            "You must strictly rely on the provided context from the vector store. "
            "If the context does not contain relevant information, DO NOT generate a response. "
            "Instead, respond with: 'I do not have the necessary context in my knowledge base. Please contact customer support for further assistance.'\n"
            "Here is the retrieved context:\n\n" + context_text
        )
    }]

    user_messages = [{"role": "user", "content": f"Here is my question/thoughts:\n{question}\n\n"}]

    prompt = conversation_history + system_messages + user_messages

    loop = asyncio.get_running_loop()
    chat_completion = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            messages=prompt, model="gpt-4o", temperature=0, max_tokens=4096
        )
    )

    answer_text = chat_completion.choices[0].message.content

    if "I do not have the necessary context" in answer_text:
        return "I do not have the necessary context in my knowledge base. Please contact customer support for further assistance."

    conversation_history.append(user_messages[0])  
    conversation_history.append({"role": "assistant", "content": answer_text})  

    return answer_text
