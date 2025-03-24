from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from http import HTTPStatus
import pandas as pd
from io import StringIO
from prompt_optimizer.helper.schema import OptimizeResponse, OptimizeFileUploadRequest
from prompt_optimizer.model import GPTModel
from prompt_optimizer.config import OPTIMIZER_CONFIG
from prompt_optimizer.prompt_optimizer import run_optimizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": HTTPStatus.OK,
    }

@app.post("/optimize")
async def optimize(request: Request):
    data = await request.json()
    return {
        "status": HTTPStatus.OK,
    }

async def get_optimize_request_form(
    system_prompt: str = Form(...),
    iterations: int = Form(None),
    chunk_size: int = Form(None),
    llm_client: str = Form(...)
) -> OptimizeFileUploadRequest:
    """
    Dependency that creates an OptimizeFileUploadRequest from form data.
    """
    return OptimizeFileUploadRequest(
        system_prompt=system_prompt,
        iterations=iterations,
        chunk_size=chunk_size,
        llm_client=llm_client
    )

@app.post("/optimize/upload", response_model=OptimizeResponse)
async def optimize_with_csv_upload(
    file: UploadFile = File(...),
    request: OptimizeFileUploadRequest = Depends(get_optimize_request_form)
):
    """
    Endpoint that accepts a CSV file upload and returns an optimized prompt.
    
    Args:
        file: Uploaded CSV file with input and ground_truth columns
        request: Request object containing optimization parameters
    
    Returns:
        JSON response with optimized prompt
    """
    # Validate file extension
    if file.content_type != "text/csv":
        return {"error": "Invalid file type. Please upload a CSV file."}

    # Read CSV file into Pandas DataFrame
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))
    config_dict = {
        "max_iterations": request.iterations if request.iterations else OPTIMIZER_CONFIG.max_iterations,
        "chunk_size": request.chunk_size if request.chunk_size else OPTIMIZER_CONFIG.chunk_size
    }
    try:
        if request.llm_client == "gpt":
            model = GPTModel()
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid LLM client. Only 'gpt' is supported for now."
            )
        optimized_prompt = await run_optimizer(llm_client=model, 
                                        initial_prompt=request.system_prompt, 
                                        input_ground_truth_csv=df,
                                        config_dict=config_dict)
                
        # Return the result
        return {
            "status": HTTPStatus.OK,
            "optimized_prompt": optimized_prompt,
            "iterations_completed": config_dict["max_iterations"]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during optimization: {str(e)}"
        )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"}
    )