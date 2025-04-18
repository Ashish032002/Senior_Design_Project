"""
This type stub file was generated by pyright.
"""

from google.oauth2 import utils

"""OAuth 2.0 Token Exchange Spec.

This module defines a token exchange utility based on the `OAuth 2.0 Token
Exchange`_ spec. This will be mainly used to exchange external credentials
for GCP access tokens in workload identity pools to access Google APIs.

The implementation will support various types of client authentication as
allowed in the spec.

A deviation on the spec will be for additional Google specific options that
cannot be easily mapped to parameters defined in the RFC.

The returned dictionary response will be based on the `rfc8693 section 2.2.1`_
spec JSON response.

.. _OAuth 2.0 Token Exchange: https://tools.ietf.org/html/rfc8693
.. _rfc8693 section 2.2.1: https://tools.ietf.org/html/rfc8693#section-2.2.1
"""
_URLENCODED_HEADERS = ...
class Client(utils.OAuthClientAuthHandler):
    """Implements the OAuth 2.0 token exchange spec based on
    https://tools.ietf.org/html/rfc8693.
    """
    def __init__(self, token_exchange_endpoint, client_authentication=...) -> None:
        """Initializes an STS client instance.

        Args:
            token_exchange_endpoint (str): The token exchange endpoint.
            client_authentication (Optional(google.oauth2.oauth2_utils.ClientAuthentication)):
                The optional OAuth client authentication credentials if available.
        """
        ...
    
    def exchange_token(self, request, grant_type, subject_token, subject_token_type, resource=..., audience=..., scopes=..., requested_token_type=..., actor_token=..., actor_token_type=..., additional_options=..., additional_headers=...): # -> Any:
        """Exchanges the provided token for another type of token based on the
        rfc8693 spec.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            grant_type (str): The OAuth 2.0 token exchange grant type.
            subject_token (str): The OAuth 2.0 token exchange subject token.
            subject_token_type (str): The OAuth 2.0 token exchange subject token type.
            resource (Optional[str]): The optional OAuth 2.0 token exchange resource field.
            audience (Optional[str]): The optional OAuth 2.0 token exchange audience field.
            scopes (Optional[Sequence[str]]): The optional list of scopes to use.
            requested_token_type (Optional[str]): The optional OAuth 2.0 token exchange requested
                token type.
            actor_token (Optional[str]): The optional OAuth 2.0 token exchange actor token.
            actor_token_type (Optional[str]): The optional OAuth 2.0 token exchange actor token type.
            additional_options (Optional[Mapping[str, str]]): The optional additional
                non-standard Google specific options.
            additional_headers (Optional[Mapping[str, str]]): The optional additional
                headers to pass to the token exchange endpoint.

        Returns:
            Mapping[str, str]: The token exchange JSON-decoded response data containing
                the requested token and its expiration time.

        Raises:
            google.auth.exceptions.OAuthError: If the token endpoint returned
                an error.
        """
        ...
    
    def refresh_token(self, request, refresh_token): # -> Any:
        """Exchanges a refresh token for an access token based on the
        RFC6749 spec.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            subject_token (str): The OAuth 2.0 refresh token.
        """
        ...
    


