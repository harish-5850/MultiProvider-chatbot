import os
from dotenv import load_dotenv
from google import genai

# 1. Load environment variables
load_dotenv()

# 2. Initialize the Modern Client
# If your .env has GOOGLE_API_KEY, the client picks it up automatically!
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    # 3. Make the request using the modern syntax
    # We'll use 'gemini-1.5-flash' (or 'gemini-2.0-flash' if available)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explain the difference between a legacy SDK and a modern SDK."
    )
   
    print("\n--- Modern Gemini Response ---")
    print(response.text)
   
except Exception as e:
    print(f"❌ Modern API Error: {e}")