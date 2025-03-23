from pydantic import BaseModel, Field, validator
from typing import Optional, Union, List
import os


class OptimizeFileUploadRequest(BaseModel):
    """
    Schema for file upload optimization requests.
    """
    # File upload is handled separately
    system_prompt: str = Field(
        ..., 
        description="Initial system prompt to optimize"
    )
    llm_client: str = Field(
        default="gpt",
        description="LLM client to use for optimization"
    )
    iterations: int = Field(
        default=1, 
        ge=1, 
        description="Number of optimization iterations to run"
    )
    chunk_size: int = Field(
        default=10, 
        ge=1, 
        description="Number of examples to process in each chunk"
    )
    
    @validator('system_prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("System prompt cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "system_prompt": "You are a helpful assistant that provides accurate historical information.",
                "iterations": 2,
                "chunk_size": 5
            }
        }


class OptimizeResponse(BaseModel):
    """
    Schema for optimization response.
    """
    optimized_prompt: str
    iterations_completed: int
    success: bool = True
    message: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "optimized_prompt": "You are an expert historical assistant...",
                "iterations_completed": 2,
                "success": True,
                "message": "Optimization completed successfully"
            }
        }
    
    