import sys
import os
import time
from dotenv import load_dotenv

# 🔥 Load environment variables once (Entry Point)
load_dotenv()

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_service import LLMFactory


def run_battle(prompt):
    providers = ["gemini"]

    print(f"⚔️ Starting Battle for Prompt: '{prompt}'\n")

    for p_type in providers:
        provider = LLMFactory.get_provider(p_type)

        start_time = time.time()
        result = provider.generate(prompt)
        duration = time.time() - start_time

        print(f"--- Provider: {p_type.upper()} ---")
        print(f"⏱️ Latency: {duration:.2f}s")
        print(f"📄 Response: {result[:120]}...")
        print("-" * 40)


if __name__ == "__main__":
    run_battle("Write a short tagline for a GenAI course at MVGR college.")
