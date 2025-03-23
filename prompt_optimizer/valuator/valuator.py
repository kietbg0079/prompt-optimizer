from typing import Dict, Any, List, Optional
import logging

from prompt_optimizer.prompt_template import VALUATOR_PROMPT
from prompt_optimizer.model import BaseModel
from .summarize_suggestions import Summarizer

class Valuator:
    """
    A class that evaluates LLM outputs against ground truth using a structured analysis prompt.
    """
    
    def __init__(self, 
                 llm_client: BaseModel):
        """
        Initialize the Valuator with a prompt template.
        
        Args:
            prompt_template_path: Path to the valuation prompt template
            llm_client: LLM client for executing the valuation (if None, will only prepare prompts)
        """
        self.llm_client = llm_client
        
    def prepare_valuation_prompt(self,
                                 system_prompt: str,
                                 input_data: str,
                                 llm_output: str,
                                 ground_truth: str) -> str:
        """
        Generate the complete valuation prompt by filling in the template.
        
        Returns:
            The complete valuation prompt with all input data
        """  
        return VALUATOR_PROMPT.format(
            system_prompt=system_prompt,
            input=input_data,
            llm_generated_output=llm_output,
            ground_truth_output=ground_truth
        )
    
    async def valuate(self,
                input_data: str,
                system_prompt: str,
                ground_truth: str,
                llm_output: str = None) -> Dict[str, Any]:
        """
        Run the valuation process using the configured LLM client.
        
        Returns:
            Dictionary containing the valuation results
        """
        if self.llm_client is None:
            raise ValueError("No LLM client provided for valuation")
        
        if llm_output == None:
            llm_output = await self.llm_client.generate_async(
                [
                    ("system", system_prompt), 
                    ("user", input_data)
                ]
            )

        prompt = self.prepare_valuation_prompt(system_prompt=system_prompt, 
                                               input_data=input_data, 
                                               llm_output=llm_output, 
                                               ground_truth=ground_truth)
        try:
            response = await self.llm_client.generate_async(prompt)
            return response
        
        except Exception as e:
            logging.error(f"Error during valuation: {e}")
            raise
    
    async def valuates(self, 
                       data_chunk: List[Dict[str, Any]],
                       system_prompt: str,
                       llm_outputs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Valuate multiple input-output pairs concurrently.
        
        Args:
            data_chunk: List of data items containing input, ground_truth, and system_prompt
            llm_outputs: Optional list of model outputs corresponding to inputs
            
        Returns:
            List of valuation results
        """
        if not data_chunk:
            return []
        
        # Create tasks for concurrent evaluation
        import asyncio
        
        tasks = []
        for i, item in enumerate(data_chunk):
            llm_output = None if llm_outputs is None else llm_outputs[i]
            task = self.valuate(
                input_data=item["input"],
                system_prompt=system_prompt,
                ground_truth=item["ground_truth"],
                llm_output=llm_output
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        try:
            suggestions = await asyncio.gather(*tasks)
            final_suggestion = Summarizer(suggestions, self.llm_client).summarize()
            return final_suggestion
        except Exception as e:
            logging.error(f"Error during batch valuation: {e}")
            raise

    def _parse_valuation_response(self, response: Any) -> Dict[str, Any]:
        pass