"""
This type stub file was generated by pyright.
"""

"""Module for classes related to identifying a Sheets document."""
class SheetsURL:
    """Class that enforces safety by ensuring that URLs are sanitized."""
    def __init__(self, url: str) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    


class SheetsKey:
    """Class that enforces safety by ensuring that keys are sanitized."""
    def __init__(self, key: str) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    


class SheetsIdentifier:
    """Encapsulates a means to identify a Sheets document.

    The gspread library provides three ways to look up a Sheets document: by name,
    by url and by key. An instance of this class represents exactly one of the
    methods.
    """
    def __init__(self, name: str | None = ..., key: SheetsKey | None = ..., url: SheetsURL | None = ...) -> None:
        """Constructor.

        Exactly one of the arguments should be provided.

        Args:
          name: The name of the Sheets document. More-than-one Sheets documents can
            have the same name, so this is the least precise method of identifying
            the document.
          key: The key of the Sheets document
          url: The url to the Sheets document

        Raises:
          ValueError: If the caller does not specify exactly one of name, url or
          key.
        """
        ...
    
    def name(self) -> str | None:
        ...
    
    def key(self) -> SheetsKey | None:
        ...
    
    def url(self) -> SheetsURL | None:
        ...
    
    def __str__(self) -> str:
        ...
    


