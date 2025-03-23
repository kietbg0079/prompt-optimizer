from typing import List
from prompt_optimizer.prompt_template import SUMMARIZE_SUGGESTIONS_PROMPT
from prompt_optimizer.model import BaseModel

class Summarizer:
    def __init__(self, 
                 llm_client: BaseModel,
                 ):
        self.llm_client = llm_client
        
    def _prepare_summarize_prompt(self, valuate_results: List[str]) -> str:
        """
        Prepare the summarize prompt.
        """
        return SUMMARIZE_SUGGESTIONS_PROMPT.format(suggestions=valuate_results)
    
    def summarize(self, valuate_results: List[str]) -> str:
        """
        Summarize the valuate results.
        """
        prompt = self._prepare_summarize_prompt(valuate_results)
        response = self.llm_client.generate(prompt)
        return response
        
        