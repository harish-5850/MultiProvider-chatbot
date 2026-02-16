import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Listing available models for your API key...\n")

# This lists every model available to you
for model in client.models.list():
    # Only show models that support 'generateContent'
    if 'generateContent' in model.supported_actions:
        print(f"-> Model ID: {model.name}")
