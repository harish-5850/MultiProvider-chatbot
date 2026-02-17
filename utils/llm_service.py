from abc import ABC, abstractmethod
import os
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import openai
import anthropic

load_dotenv()

# 1. The Contract (The Interface)
class BaseProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str):
        """Yields chunks of text as they arrive"""
        pass


# 2. The Gemini Implementation
class GeminiProvider(BaseProvider):
    def __init__(self, model_id="gemini-2.5-flash"):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_id = model_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_stream(self, prompt: str):
        # The new SDK uses 'models.generate_content_stream'
        for chunk in self.client.models.generate_content_stream(
            model=self.model_id,
            contents=prompt
        ):
            if chunk.text:
                yield chunk.text

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        return response.text

# 3. OpenAI Provider
class OpenAIProvider(BaseProvider):
    def __init__(self, model_id="gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_id = model_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def generate_stream(self, prompt: str):
        pass # Placeholder to satisfy abstract base class

# 4. Anthropic Provider
class AnthropicProvider(BaseProvider):
    def __init__(self, model_id="claude-3-haiku-20240307"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_id = model_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def generate_stream(self, prompt: str):
        pass # Placeholder to satisfy abstract base class

# 5. The Factory (How we switch providers)
class LLMFactory:
    @staticmethod
    def get_provider(provider_type: str) -> BaseProvider:
        if provider_type == "gemini":
            return GeminiProvider()
        elif provider_type == "openai":
            return OpenAIProvider()
        elif provider_type == "anthropic":
            return AnthropicProvider()
        raise ValueError(f"Provider {provider_type} not supported.")