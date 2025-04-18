"""
This type stub file was generated by pyright.
"""

import abc
import http.client as http_client

"""
This type stub file was generated by pyright.
"""
DEFAULT_RETRYABLE_STATUS_CODES = ...
DEFAULT_REFRESH_STATUS_CODES = ...
DEFAULT_MAX_REFRESH_ATTEMPTS = ...
class Response(metaclass=abc.ABCMeta):
    """HTTP Response data."""
    @abc.abstractproperty
    def status(self):
        """int: The HTTP status code."""
        ...
    
    @abc.abstractproperty
    def headers(self):
        """Mapping[str, str]: The HTTP response headers."""
        ...
    
    @abc.abstractproperty
    def data(self):
        """bytes: The response body."""
        ...
    


class Request(metaclass=abc.ABCMeta):
    """Interface for a callable that makes HTTP requests.

    Specific transport implementations should provide an implementation of
    this that adapts their specific request / response API.

    .. automethod:: __call__
    """
    @abc.abstractmethod
    def __call__(self, url, method=..., body=..., headers=..., timeout=..., **kwargs):
        """Make an HTTP request.

        Args:
            url (str): The URI to be requested.
            method (str): The HTTP method to use for the request. Defaults
                to 'GET'.
            body (bytes): The payload / body in HTTP request.
            headers (Mapping[str, str]): Request headers.
            timeout (Optional[int]): The number of seconds to wait for a
                response from the server. If not specified or if None, the
                transport-specific default timeout will be used.
            kwargs: Additionally arguments passed on to the transport's
                request method.

        Returns:
            Response: The HTTP response.

        Raises:
            google.auth.exceptions.TransportError: If any exception occurred.
        """
        ...
    


