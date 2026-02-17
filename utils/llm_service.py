from abc import ABC, abstractmethod
import os
import google.genai as genai
import openai
import anthropic
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()


# ======================================================
# 1️⃣ Base Interface (Contract)
# ======================================================
class BaseProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate full response"""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str):
        """Stream response chunks"""
        pass


# ======================================================
# 2️⃣ Gemini Provider
# ======================================================
class GeminiProvider(BaseProvider):

    def __init__(self, model_id="gemini-2.5-flash-lite"):
        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )
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

    def generate_stream(self, prompt: str):
        for chunk in self.client.models.generate_content_stream(
            model=self.model_id,
            contents=prompt
        ):
            if chunk.text:
                yield chunk.text


# ======================================================
# 3️⃣ OpenAI Provider
# ======================================================
class OpenAIProvider(BaseProvider):

    def __init__(self, model_id="gpt-4o-mini"):
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model_id = model_id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def generate_stream(self, prompt: str):
        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ======================================================
# 4️⃣ Anthropic Provider
# ======================================================
class AnthropicProvider(BaseProvider):

    def __init__(self, model_id="claude-3-haiku-20240307"):
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model_id = model_id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def generate_stream(self, prompt: str):
        with self.client.messages.stream(
            model=self.model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text


# ======================================================
# 5️⃣ Factory Pattern
# ======================================================
class LLMFactory:

    @staticmethod
    def get_provider(provider_type: str) -> BaseProvider:

        if provider_type == "gemini":
            return GeminiProvider()

        elif provider_type == "openai":
            return OpenAIProvider()

        elif provider_type == "anthropic":
            return AnthropicProvider()

        else:
            raise ValueError(f"Provider {provider_type} not supported.")
