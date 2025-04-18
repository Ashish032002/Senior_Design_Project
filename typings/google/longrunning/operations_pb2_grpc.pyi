"""
This type stub file was generated by pyright.
"""

"""Client and server classes corresponding to protobuf-defined services."""
class OperationsStub:
    """Manages long-running operations with an API service.

    When an API method normally takes long time to complete, it can be designed
    to return [Operation][google.longrunning.Operation] to the client, and the client can use this
    interface to receive the real response asynchronously by polling the
    operation resource, or pass the operation resource to another API (such as
    Google Cloud Pub/Sub API) to receive the response.  Any API service that
    returns long-running operations should implement the `Operations` interface
    so developers can have a consistent client experience.
    """
    def __init__(self, channel) -> None:
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        ...
    


class OperationsServicer:
    """Manages long-running operations with an API service.

    When an API method normally takes long time to complete, it can be designed
    to return [Operation][google.longrunning.Operation] to the client, and the client can use this
    interface to receive the real response asynchronously by polling the
    operation resource, or pass the operation resource to another API (such as
    Google Cloud Pub/Sub API) to receive the response.  Any API service that
    returns long-running operations should implement the `Operations` interface
    so developers can have a consistent client experience.
    """
    def ListOperations(self, request, context):
        """Lists operations that match the specified filter in the request. If the
        server doesn't support this method, it returns `UNIMPLEMENTED`.

        NOTE: the `name` binding allows API services to override the binding
        to use different resource name schemes, such as `users/*/operations`. To
        override the binding, API services can add a binding such as
        `"/v1/{name=users/*}/operations"` to their service configuration.
        For backwards compatibility, the default name includes the operations
        collection id, however overriding users must ensure the name binding
        is the parent resource, without the operations collection id.
        """
        ...
    
    def GetOperation(self, request, context):
        """Gets the latest state of a long-running operation.  Clients can use this
        method to poll the operation result at intervals as recommended by the API
        service.
        """
        ...
    
    def DeleteOperation(self, request, context):
        """Deletes a long-running operation. This method indicates that the client is
        no longer interested in the operation result. It does not cancel the
        operation. If the server doesn't support this method, it returns
        `google.rpc.Code.UNIMPLEMENTED`.
        """
        ...
    
    def CancelOperation(self, request, context):
        """Starts asynchronous cancellation on a long-running operation.  The server
        makes a best effort to cancel the operation, but success is not
        guaranteed.  If the server doesn't support this method, it returns
        `google.rpc.Code.UNIMPLEMENTED`.  Clients can use
        [Operations.GetOperation][google.longrunning.Operations.GetOperation] or
        other methods to check whether the cancellation succeeded or whether the
        operation completed despite cancellation. On successful cancellation,
        the operation is not deleted; instead, it becomes an operation with
        an [Operation.error][google.longrunning.Operation.error] value with a [google.rpc.Status.code][google.rpc.Status.code] of 1,
        corresponding to `Code.CANCELLED`.
        """
        ...
    
    def WaitOperation(self, request, context):
        """Waits until the specified long-running operation is done or reaches at most
        a specified timeout, returning the latest state.  If the operation is
        already done, the latest state is immediately returned.  If the timeout
        specified is greater than the default HTTP/RPC timeout, the HTTP/RPC
        timeout is used.  If the server does not support this method, it returns
        `google.rpc.Code.UNIMPLEMENTED`.
        Note that this method is on a best-effort basis.  It may return the latest
        state before the specified timeout (including immediately), meaning even an
        immediate response is no guarantee that the operation is done.
        """
        ...
    


def add_OperationsServicer_to_server(servicer, server): # -> None:
    ...

class Operations:
    """Manages long-running operations with an API service.

    When an API method normally takes long time to complete, it can be designed
    to return [Operation][google.longrunning.Operation] to the client, and the client can use this
    interface to receive the real response asynchronously by polling the
    operation resource, or pass the operation resource to another API (such as
    Google Cloud Pub/Sub API) to receive the response.  Any API service that
    returns long-running operations should implement the `Operations` interface
    so developers can have a consistent client experience.
    """
    @staticmethod
    def ListOperations(request, target, options=..., channel_credentials=..., call_credentials=..., insecure=..., compression=..., wait_for_ready=..., timeout=..., metadata=...):
        ...
    
    @staticmethod
    def GetOperation(request, target, options=..., channel_credentials=..., call_credentials=..., insecure=..., compression=..., wait_for_ready=..., timeout=..., metadata=...):
        ...
    
    @staticmethod
    def DeleteOperation(request, target, options=..., channel_credentials=..., call_credentials=..., insecure=..., compression=..., wait_for_ready=..., timeout=..., metadata=...):
        ...
    
    @staticmethod
    def CancelOperation(request, target, options=..., channel_credentials=..., call_credentials=..., insecure=..., compression=..., wait_for_ready=..., timeout=..., metadata=...):
        ...
    
    @staticmethod
    def WaitOperation(request, target, options=..., channel_credentials=..., call_credentials=..., insecure=..., compression=..., wait_for_ready=..., timeout=..., metadata=...):
        ...
    


