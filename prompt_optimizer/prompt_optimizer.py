from typing import List, Tuple, Union
from pandas import DataFrame

from prompt_optimizer.model import BaseModel, GPTModel
from prompt_optimizer.rewriter import Rewriter
from prompt_optimizer.valuator import Valuator, Summarizer
from prompt_optimizer.helper.dataloader import DataLoader
from prompt_optimizer.helper.utils import run_async
from prompt_optimizer.config import OPTIMIZER_CONFIG

class PromptOptimizer:
    def __init__(self, 
                 llm_client: BaseModel,
                 config_dict: dict = {}):
        self.llm_client = llm_client
        self.valuator = Valuator(self.llm_client)
        self.rewriter = Rewriter(self.llm_client)
        self.summarizer = Summarizer(self.llm_client)
        self._load_config(config_dict)

    def _load_config(self, config_dict: dict) -> None:
        self.max_iterations = config_dict.get("max_iterations", OPTIMIZER_CONFIG.max_iterations)
        self.chunk_size = config_dict.get("chunk_size", OPTIMIZER_CONFIG.chunk_size)

    async def optimize(self, 
                 input_ground_truth_csv: Union[str, DataFrame], 
                 initial_system_prompt: str) -> str:
        
        # Load the data
        data_loader = DataLoader(input_ground_truth_csv)
        suggestions = []
        for chunk in data_loader.get_chunks(self.chunk_size):
            # Rewrite the prompt
            suggestion = await self.valuator.valuates(chunk, initial_system_prompt)
            suggestions.append(suggestion)
        
        # Summarize the suggestions
        final_suggestion = self.summarizer.summarize(suggestions)
        prompt_rewrite = self.rewriter.rewrite(initial_system_prompt, final_suggestion)
        return prompt_rewrite
        
    async def run(self, 
            input_ground_truth_csv: str, 
            initial_system_prompt: str) -> str:
        """
        Run the prompt optimizer.
        """
        optimized_prompt = initial_system_prompt
        for iter in range(self.max_iterations):
            optimized_prompt = await self.optimize(input_ground_truth_csv, optimized_prompt)
            print(f"Iteration {iter+1}: {optimized_prompt}")

        return optimized_prompt

async def run_optimizer(llm_client: BaseModel,
                  initial_prompt: str, 
                  input_ground_truth_csv: Union[str, DataFrame],
                  config_dict: dict = {}) -> str:
    optimizer = PromptOptimizer(llm_client, config_dict)
    optimized_prompt = await optimizer.run(
        input_ground_truth_csv=input_ground_truth_csv,
        initial_system_prompt=initial_prompt
    )
    return optimized_prompt

if __name__ == "__main__":
    initial_prompt = """You are an educational assistant specializing in historical facts and events. When presented with questions about history, provide accurate, concise, and informative responses based on established historical consensus.

Follow these guidelines:
1. Provide factually accurate information, including relevant dates, people, and contexts.
2. Maintain objectivity when discussing controversial historical topics.
3. Include the most important details while keeping responses clear and accessible.
4. Acknowledge when historical interpretations vary or when evidence is inconclusive.
5. Avoid using overly technical language unless necessary for accuracy.

Your goal is to help users understand historical events, their significance, and their connections to broader historical developments."""

    optimizer = PromptOptimizer(GPTModel())
    optimized_prompt = optimizer.run(
        input_ground_truth_csv="sample_data.csv",
        initial_system_prompt=initial_prompt
    )
    print(optimized_prompt)