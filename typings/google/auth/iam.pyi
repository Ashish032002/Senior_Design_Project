"""
This type stub file was generated by pyright.
"""

from google.auth import _helpers, crypt

"""Tools for using the Google `Cloud Identity and Access Management (IAM)
API`_'s auth-related functionality.

.. _Cloud Identity and Access Management (IAM) API:
    https://cloud.google.com/iam/docs/
"""
IAM_RETRY_CODES = ...
_IAM_SCOPE = ...
_IAM_ENDPOINT = ...
_IAM_SIGN_ENDPOINT = ...
_IAM_IDTOKEN_ENDPOINT = ...
class Signer(crypt.Signer):
    """Signs messages using the IAM `signBlob API`_.

    This is useful when you need to sign bytes but do not have access to the
    credential's private key file.

    .. _signBlob API:
        https://cloud.google.com/iam/reference/rest/v1/projects.serviceAccounts
        /signBlob
    """
    def __init__(self, request, credentials, service_account_email) -> None:
        """
        Args:
            request (google.auth.transport.Request): The object used to make
                HTTP requests.
            credentials (google.auth.credentials.Credentials): The credentials
                that will be used to authenticate the request to the IAM API.
                The credentials must have of one the following scopes:

                - https://www.googleapis.com/auth/iam
                - https://www.googleapis.com/auth/cloud-platform
            service_account_email (str): The service account email identifying
                which service account to use to sign bytes. Often, this can
                be the same as the service account email in the given
                credentials.
        """
        ...
    
    @property
    def key_id(self): # -> None:
        """Optional[str]: The key ID used to identify this private key.

        .. warning::
           This is always ``None``. The key ID used by IAM can not
           be reliably determined ahead of time.
        """
        ...
    
    @_helpers.copy_docstring(crypt.Signer)
    def sign(self, message): # -> bytes:
        ...
    


