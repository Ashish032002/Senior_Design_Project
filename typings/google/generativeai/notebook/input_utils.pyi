"""
This type stub file was generated by pyright.
"""

from google.generativeai.notebook import parsed_args_lib
from google.generativeai.notebook.lib import llmfn_inputs_source

"""Utilities for handling input variables."""
class _NormalizedInputsSource(llmfn_inputs_source.LLMFnInputsSource):
    """Wrapper around NormalizedInputsList.

    By design LLMFunction does not take NormalizedInputsList as input because
    NormalizedInputsList is an internal representation so we want to minimize
    exposure to the caller.

    When we have inputs already in normalized format (e.g. from
    join_prompt_inputs()) we can wrap it as an LLMFnInputsSource to pass as an
    input to LLMFunction.
    """
    def __init__(self, normalized_inputs: llmfn_inputs_source.NormalizedInputsList) -> None:
        ...
    


def get_inputs_source_from_py_var(var_name: str) -> llmfn_inputs_source.LLMFnInputsSource:
    ...

def join_inputs_sources(parsed_args: parsed_args_lib.ParsedArgs, suppress_status_msgs: bool = ...) -> llmfn_inputs_source.LLMFnInputsSource:
    """Get a single combined input source from `parsed_args."""
    ...

