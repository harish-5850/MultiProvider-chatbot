import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_service import LLMFactory
import time

def run_stream():
    provider = LLMFactory.get_provider("gemini")
    prompt = "Write a 200-word creative story about a robot learning to paint at MVGR college."
    
    print("🤖 Gemini is thinking...\n")
    
    print("--- STREAM START ---")
    for chunk in provider.generate_stream(prompt):
        # end="" prevents newlines, flush=True forces it to the screen immediately
        print(chunk, end="", flush=True)
    print("\n--- STREAM END ---")

if __name__ == "__main__":
    run_stream()
