from anthropic import Anthropic

from app.llm.base import LLMClient


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-opus-5-5", max_tokens: int = 1024):
        self._client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
