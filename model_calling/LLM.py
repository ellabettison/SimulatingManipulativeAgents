from abc import ABC, abstractmethod

class LLM(ABC):
    model = None
    @abstractmethod
    def chat(self, user_prompt:str, system_prompt:str=None, max_tokens: int=500) -> str:
        pass