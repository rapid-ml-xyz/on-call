from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


class LLMFactory:
    """Factory class for creating LLM instances"""

    @staticmethod
    def open_ai_llm(model_name: str = 'gpt-4o-mini', temperature: float = 0.7) -> ChatOpenAI:
        return ChatOpenAI(model=model_name, temperature=temperature)

    @staticmethod
    def anthropic_llm(model_name: str = 'claude-3-sonnet-20240229', temperature: float = 0.7) -> ChatAnthropic:
        return ChatAnthropic(model=model_name, temperature=temperature)
