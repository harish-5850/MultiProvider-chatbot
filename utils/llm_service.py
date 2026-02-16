from abc import ABC, abstractmethod
import os
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential


# 1️⃣ Base Provider Contract
class BaseProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


# 2️⃣ Gemini Provider
class GeminiProvider(BaseProvider):

    def __init__(self, model_id="gemini-2.5-flash-lite"):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_id = model_id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        return response.text


# 3️⃣ Factory Pattern
class LLMFactory:

    @staticmethod
    def get_provider(provider_type: str) -> BaseProvider:

        if provider_type == "gemini":
            return GeminiProvider()

        raise ValueError(f"Provider {provider_type} not supported.")
