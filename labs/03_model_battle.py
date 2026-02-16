from utils.llm_service import LLMFactory
import time

def run_battle(prompt):
    # We can easily add "openai" here once we have a key!
    providers = ["gemini"] 
    
    print(f"⚔️ Starting Battle for Prompt: '{prompt}'\n")
    
    for p_type in providers:
        provider = LLMFactory.get_provider(p_type)
        
        start_time = time.time()
        result = provider.generate(prompt)
        duration = time.time() - start_time
        
        print(f"--- Provider: {p_type.upper()} ---")
        print(f"⏱️ Latency: {duration:.2f}s")
        print(f"📄 Response: {result[:100]}...") # Print first 100 chars
        print("-" * 30)

if __name__ == "__main__":
    run_battle("Write a short tagline for a GenAI course at MVGR college.")
