from openai import OpenAI

from app.llm.base import LLMClient


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 1024):
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
