import asyncio

def run_async(async_func, *args, **kwargs):
    """
    Run an async function from a synchronous context.
    
    Args:
        async_func: Async function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the async function
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_func(*args, **kwargs))