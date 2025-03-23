import openai
from openai import AsyncOpenAI
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from .base_model import BaseModel, run_async
from prompt_optimizer.config import LLM_CONFIG

class GPTModel(BaseModel):
    def __init__(self):
        init_params = LLM_CONFIG.__dict__
        LLM_CONFIG.__dict__.pop("provider")
        super().__init__(
            **init_params
        )
        self._initialize_async_client()
    
    def _initialize_client(self):
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def _initialize_async_client(self):
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    def _process_messages(self, raw_messages) -> List[Dict[str, str]]:
        messages = []
        if isinstance(raw_messages, str):
            messages = [{"role": "user", "content": raw_messages}]
        elif isinstance(raw_messages, list):
            if all(isinstance(message, dict) for message in raw_messages):
                messages = raw_messages
            elif all(isinstance(message, tuple) and len(message) == 2 for message in raw_messages):
                messages = [{"role": message[0], "content": message[1]} for message in raw_messages]
            else:
                raise ValueError("Invalid messages format")
        else:
            raise ValueError("Invalid messages format")

        return messages
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        messages = self._process_messages(messages)
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            **kwargs
        }

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        def _generate():
            response = self.client.chat.completions.create(
                messages=messages,
                **params
            )
            return response.choices[0].message.content
        
        return self.with_retries(_generate)

    async def generate_async(self, messages: List[Dict[str, str]], **kwargs) -> str:
        messages = self._process_messages(messages)
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            **kwargs
        }
        
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        async def _generate_async():
            response = await self.async_client.chat.completions.create(
                messages=messages,
                **params
            )   
            return response.choices[0].message.content
        
        return await self.with_retries_async(_generate_async)
    

if __name__ == "__main__":
    model = GPTModel()
    print(model.generate([{"role": "user", "content": "Hello, how are you?"}]))

