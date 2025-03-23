from pathlib import Path

with (Path(__file__).parents[0] / Path('valuator_prompt.md')).open('r') as f:
    VALUATOR_PROMPT = f.read()

with (Path(__file__).parents[0] / Path('rewriter_prompt.md')).open('r') as f:
    REWRITER_PROMPT = f.read()

with (Path(__file__).parents[0] / Path('summarize_suggestions.md')).open('r') as f:
    SUMMARIZE_SUGGESTIONS_PROMPT = f.read()

__all__ = ["VALUATOR_PROMPT", "REWRITER_PROMPT", "SUMMARIZE_SUGGESTIONS_PROMPT"]

