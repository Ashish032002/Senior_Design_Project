"""
This type stub file was generated by pyright.
"""

from typing import Any
from google.generativeai.notebook import ipython_env

"""IPythonEnvImpl."""
class IPythonEnvImpl(ipython_env.IPythonEnv):
    """Concrete implementation of IPythonEnv."""
    def display(self, x: Any) -> None:
        ...
    
    def display_html(self, x: str) -> None:
        ...
    


