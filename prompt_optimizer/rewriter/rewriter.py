import logging

from prompt_optimizer.model import BaseModel
from prompt_optimizer.prompt_template import REWRITER_PROMPT

class Rewriter:
    def __init__(self, 
                 llm_client: BaseModel):
        self.llm_client = llm_client

    def prepare_rewriter_prompt(self, 
                                original_system_prompt: str, 
                                prompt_suggestion: str) -> str:
        return REWRITER_PROMPT.format(
            original_system_prompt=original_system_prompt,
            prompt_suggestion=prompt_suggestion
        )

    def rewrite(self, 
                original_system_prompt: str, 
                prompt_suggestion: str) -> str:
        prompt = self.prepare_rewriter_prompt(original_system_prompt, prompt_suggestion)

        try:
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logging.error(f"Error during rewriting: {e}")
            raise
