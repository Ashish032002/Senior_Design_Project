"""
This type stub file was generated by pyright.
"""

"""
This type stub file was generated by pyright.
"""
class OperationsClient:
    """Client for interacting with long-running operations within a service.

    Args:
        channel (grpc.Channel): The gRPC channel associated with the service
            that implements the ``google.longrunning.operations`` interface.
        client_config (dict):
            A dictionary of call options for each method. If not specified
            the default configuration is used.
    """
    def __init__(self, channel, client_config=...) -> None:
        ...
    
    def get_operation(self, name, retry=..., timeout=..., compression=..., metadata=...):
        """Gets the latest state of a long-running operation.

        Clients can use this method to poll the operation result at intervals
        as recommended by the API service.

        Example:
            >>> from google.api_core import operations_v1
            >>> api = operations_v1.OperationsClient()
            >>> name = ''
            >>> response = api.get_operation(name)

        Args:
            name (str): The name of the operation resource.
            retry (google.api_core.retry.Retry): The retry strategy to use
                when invoking the RPC. If unspecified, the default retry from
                the client configuration will be used. If ``None``, then this
                method will not retry the RPC at all.
            timeout (float): The amount of time in seconds to wait for the RPC
                to complete. Note that if ``retry`` is used, this timeout
                applies to each individual attempt and the overall time it
                takes for this method to complete may be longer. If
                unspecified, the the default timeout in the client
                configuration is used. If ``None``, then the RPC method will
                not time out.
            compression (grpc.Compression): An element of grpc.compression
                e.g. grpc.compression.Gzip.
            metadata (Optional[List[Tuple[str, str]]]):
                Additional gRPC metadata.

        Returns:
            google.longrunning.operations_pb2.Operation: The state of the
                operation.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If an error occurred
                while invoking the RPC, the appropriate ``GoogleAPICallError``
                subclass will be raised.
        """
        ...
    
    def list_operations(self, name, filter_, retry=..., timeout=..., compression=..., metadata=...):
        """
        Lists operations that match the specified filter in the request.

        Example:
            >>> from google.api_core import operations_v1
            >>> api = operations_v1.OperationsClient()
            >>> name = ''
            >>>
            >>> # Iterate over all results
            >>> for operation in api.list_operations(name):
            >>>   # process operation
            >>>   pass
            >>>
            >>> # Or iterate over results one page at a time
            >>> iter = api.list_operations(name)
            >>> for page in iter.pages:
            >>>   for operation in page:
            >>>     # process operation
            >>>     pass

        Args:
            name (str): The name of the operation collection.
            filter_ (str): The standard list filter.
            retry (google.api_core.retry.Retry): The retry strategy to use
                when invoking the RPC. If unspecified, the default retry from
                the client configuration will be used. If ``None``, then this
                method will not retry the RPC at all.
            timeout (float): The amount of time in seconds to wait for the RPC
                to complete. Note that if ``retry`` is used, this timeout
                applies to each individual attempt and the overall time it
                takes for this method to complete may be longer. If
                unspecified, the the default timeout in the client
                configuration is used. If ``None``, then the RPC method will
                not time out.
            compression (grpc.Compression): An element of grpc.compression
                e.g. grpc.compression.Gzip.
            metadata (Optional[List[Tuple[str, str]]]): Additional gRPC
                metadata.

        Returns:
            google.api_core.page_iterator.Iterator: An iterator that yields
                :class:`google.longrunning.operations_pb2.Operation` instances.

        Raises:
            google.api_core.exceptions.MethodNotImplemented: If the server
                does not support this method. Services are not required to
                implement this method.
            google.api_core.exceptions.GoogleAPICallError: If an error occurred
                while invoking the RPC, the appropriate ``GoogleAPICallError``
                subclass will be raised.
        """
        ...
    
    def cancel_operation(self, name, retry=..., timeout=..., compression=..., metadata=...):
        """Starts asynchronous cancellation on a long-running operation.

        The server makes a best effort to cancel the operation, but success is
        not guaranteed. Clients can use :meth:`get_operation` or service-
        specific methods to check whether the cancellation succeeded or whether
        the operation completed despite cancellation. On successful
        cancellation, the operation is not deleted; instead, it becomes an
        operation with an ``Operation.error`` value with a
        ``google.rpc.Status.code`` of ``1``, corresponding to
        ``Code.CANCELLED``.

        Example:
            >>> from google.api_core import operations_v1
            >>> api = operations_v1.OperationsClient()
            >>> name = ''
            >>> api.cancel_operation(name)

        Args:
            name (str): The name of the operation resource to be cancelled.
            retry (google.api_core.retry.Retry): The retry strategy to use
                when invoking the RPC. If unspecified, the default retry from
                the client configuration will be used. If ``None``, then this
                method will not retry the RPC at all.
            timeout (float): The amount of time in seconds to wait for the RPC
                to complete. Note that if ``retry`` is used, this timeout
                applies to each individual attempt and the overall time it
                takes for this method to complete may be longer. If
                unspecified, the the default timeout in the client
                configuration is used. If ``None``, then the RPC method will
                not time out.
            compression (grpc.Compression): An element of grpc.compression
                e.g. grpc.compression.Gzip.
            metadata (Optional[List[Tuple[str, str]]]): Additional gRPC
                metadata.

        Raises:
            google.api_core.exceptions.MethodNotImplemented: If the server
                does not support this method. Services are not required to
                implement this method.
            google.api_core.exceptions.GoogleAPICallError: If an error occurred
                while invoking the RPC, the appropriate ``GoogleAPICallError``
                subclass will be raised.
        """
        ...
    
    def delete_operation(self, name, retry=..., timeout=..., compression=..., metadata=...):
        """Deletes a long-running operation.

        This method indicates that the client is no longer interested in the
        operation result. It does not cancel the operation.

        Example:
            >>> from google.api_core import operations_v1
            >>> api = operations_v1.OperationsClient()
            >>> name = ''
            >>> api.delete_operation(name)

        Args:
            name (str): The name of the operation resource to be deleted.
            retry (google.api_core.retry.Retry): The retry strategy to use
                when invoking the RPC. If unspecified, the default retry from
                the client configuration will be used. If ``None``, then this
                method will not retry the RPC at all.
            timeout (float): The amount of time in seconds to wait for the RPC
                to complete. Note that if ``retry`` is used, this timeout
                applies to each individual attempt and the overall time it
                takes for this method to complete may be longer. If
                unspecified, the the default timeout in the client
                configuration is used. If ``None``, then the RPC method will
                not time out.
            compression (grpc.Compression): An element of grpc.compression
                e.g. grpc.compression.Gzip.
            metadata (Optional[List[Tuple[str, str]]]): Additional gRPC
                metadata.

        Raises:
            google.api_core.exceptions.MethodNotImplemented: If the server
                does not support this method. Services are not required to
                implement this method.
            google.api_core.exceptions.GoogleAPICallError: If an error occurred
                while invoking the RPC, the appropriate ``GoogleAPICallError``
                subclass will be raised.
        """
        ...
    


