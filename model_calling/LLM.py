from abc import ABC, abstractmethod
import logging
class LLM(ABC):
    model = None

    @abstractmethod
    def call_model(self, user_prompt:str, system_prompt:str=None, max_tokens: int=1_000) -> str:
        pass
    
    def chat(self, user_prompt:str, system_prompt:str=None, max_tokens: int=1_000) -> str:
        logging.getLogger().info(f"Calling model: {self.model}")
        logging.getLogger().info(f"System prompt: {system_prompt}")
        logging.getLogger().info(f"User prompt: {user_prompt}")
        response = self.call_model(user_prompt, system_prompt, max_tokens)
        logging.getLogger().info(f"Response: {response}")
        return response