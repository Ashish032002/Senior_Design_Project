"""
This type stub file was generated by pyright.
"""

import dataclasses
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from google.api_core import gapic_v1
from google.auth import credentials as ga_credentials
from google.auth.transport.requests import AuthorizedSession
from google.ai.generativelanguage_v1beta.types import prediction_service
from .base import PredictionServiceTransport

OptionalRetry = ...
DEFAULT_CLIENT_INFO = ...
class PredictionServiceRestInterceptor:
    """Interceptor for PredictionService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the PredictionServiceRestTransport.

    .. code-block:: python
        class MyCustomPredictionServiceInterceptor(PredictionServiceRestInterceptor):
            def pre_predict(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_predict(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = PredictionServiceRestTransport(interceptor=MyCustomPredictionServiceInterceptor())
        client = PredictionServiceClient(transport=transport)


    """
    def pre_predict(self, request: prediction_service.PredictRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[prediction_service.PredictRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for predict

        Override in a subclass to manipulate the request or metadata
        before they are sent to the PredictionService server.
        """
        ...
    
    def post_predict(self, response: prediction_service.PredictResponse) -> prediction_service.PredictResponse:
        """Post-rpc interceptor for predict

        Override in a subclass to manipulate the response
        after it is returned by the PredictionService server but before
        it is returned to user code.
        """
        ...
    


@dataclasses.dataclass
class PredictionServiceRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: PredictionServiceRestInterceptor
    ...


class PredictionServiceRestTransport(PredictionServiceTransport):
    """REST backend transport for PredictionService.

    A service for online predictions and explanations.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1

    """
    def __init__(self, *, host: str = ..., credentials: Optional[ga_credentials.Credentials] = ..., credentials_file: Optional[str] = ..., scopes: Optional[Sequence[str]] = ..., client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = ..., quota_project_id: Optional[str] = ..., client_info: gapic_v1.client_info.ClientInfo = ..., always_use_jwt_access: Optional[bool] = ..., url_scheme: str = ..., interceptor: Optional[PredictionServiceRestInterceptor] = ..., api_audience: Optional[str] = ...) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'generativelanguage.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.

            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Callable[[], Tuple[bytes, bytes]]): Client
                certificate to configure mutual TLS HTTP channel. It is ignored
                if ``channel`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you are developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
            url_scheme: the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
        """
        ...
    
    class _Predict(PredictionServiceRestStub):
        def __hash__(self) -> int:
            ...
        
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = ...
        def __call__(self, request: prediction_service.PredictRequest, *, retry: OptionalRetry = ..., timeout: Optional[float] = ..., metadata: Sequence[Tuple[str, str]] = ...) -> prediction_service.PredictResponse:
            r"""Call the predict method over HTTP.

            Args:
                request (~.prediction_service.PredictRequest):
                    The request object. Request message for
                [PredictionService.Predict][google.ai.generativelanguage.v1beta.PredictionService.Predict].
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.prediction_service.PredictResponse:
                    Response message for [PredictionService.Predict].
            """
            ...
        
    
    
    @property
    def predict(self) -> Callable[[prediction_service.PredictRequest], prediction_service.PredictResponse]:
        ...
    
    @property
    def kind(self) -> str:
        ...
    
    def close(self): # -> None:
        ...
    


__all__ = ("PredictionServiceRestTransport", )
