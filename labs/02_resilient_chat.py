import os
from dotenv import load_dotenv
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# This DECORATOR makes the function resilient
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=lambda retry_state: print(f"⚠️ Rate limited. Retrying in {retry_state.next_action.sleep}s...")
)
def safe_generate(prompt):
    return client.models.generate_content(
        model="gemini-2.5-flash-lite", # Using the Lite version for maximum speed
        contents=prompt
    )

print("🚀 Starting Resilient Request...")
try:
    response = safe_generate("Write a 3-line poem about MVGR College.")
    print("\nResult:", response.text)
except Exception as e:
    print(f"\n❌ Permanent Failure: {e}")
