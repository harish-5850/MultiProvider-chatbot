import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# The list of models we want to "test" for quota
models_to_test = [
    "gemini-2.0-flash",
    "gemini-2.5-flash", 
    "gemini-2.5-flash-lite"
]

print("🔍 Testing Model Availability...\n")

for model_id in models_to_test:
    try:
        response = client.models.generate_content(
            model=model_id,
            contents="hi"
        )
        print(f"✅ {model_id}: AVAILABLE")
    except Exception as e:
        if "429" in str(e):
            print(f"❌ {model_id}: QUOTA EXHAUSTED (429)")
        else:
            print(f"❓ {model_id}: OTHER ERROR ({e})")