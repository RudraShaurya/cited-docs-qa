from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Generate a completion. Implementations use temperature 0."""
        raise NotImplementedError


class FakeLLM(LLMClient):
    """Deterministic stand-in for tests. Returns a scripted response, or
    echoes the user prompt with a trivial citation if none is scripted."""

    def __init__(self, response: str | None = None):
        self.response = response
        self.last_system: str | None = None
        self.last_user: str | None = None

    def generate(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        if self.response is not None:
            return self.response
        return "This is a test answer. [S1]"
