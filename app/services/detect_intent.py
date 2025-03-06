import os
import asyncio
from openai import AzureOpenAI  # Ensure OpenAI client is properly configured
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

async def detect_intent(user_input: str) -> str:
    """
    Uses GPT to classify user input into an intent category.
    """
    prompt = [
        {"role": "system", "content": (
            "You are an AI assistant that detects user intent. "
            "Classify the given input into one of these categories:\n"
            "- 'appointment_booking': If the user wants to book an appointment, see/consult a doctor/physician/specialist, book a medical visit.\n"
            "- 'general_question': If the user is asking a health-related question.\n"
            "- 'unclear': If the input is a greeting, small talk, or ambiguous.\n"
            "Respond with only one of these three labels and nothing else."
        )},
        {"role": "user", "content": f"User input: {user_input}"}
    ]

    loop = asyncio.get_running_loop()
    completion = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            messages=prompt,
            model="gpt-4o",
            temperature=0,
            max_tokens=10
        )
    )

    intent = completion.choices[0].message.content.strip().lower()

    # Ensure valid output
    if intent not in ["appointment_booking", "general_question", "unclear"]:
        return "unclear"
    
    return intent
