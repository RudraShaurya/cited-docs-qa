from app.config import settings
from app.llm.base import LLMClient


def get_llm_client() -> LLMClient:
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from app.llm.ollama import OllamaClient

        return OllamaClient(base_url=settings.ollama_url, model=settings.llm_model)

    if provider == "anthropic":
        from app.llm.anthropic_client import AnthropicClient

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return AnthropicClient(api_key=settings.anthropic_api_key, model=settings.llm_model)

    if provider == "openai":
        from app.llm.openai_client import OpenAIClient

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAIClient(api_key=settings.openai_api_key, model=settings.llm_model)

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
