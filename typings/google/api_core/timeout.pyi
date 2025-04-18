"""
This type stub file was generated by pyright.
"""

"""
This type stub file was generated by pyright.
"""
_DEFAULT_INITIAL_TIMEOUT = ...
_DEFAULT_MAXIMUM_TIMEOUT = ...
_DEFAULT_TIMEOUT_MULTIPLIER = ...
_DEFAULT_DEADLINE = ...
class TimeToDeadlineTimeout:
    """A decorator that decreases timeout set for an RPC based on how much time
    has left till its deadline. The deadline is calculated as
    ``now + initial_timeout`` when this decorator is first called for an rpc.

    In other words this decorator implements deadline semantics in terms of a
    sequence of decreasing timeouts t0 > t1 > t2 ... tn >= 0.

    Args:
        timeout (Optional[float]): the timeout (in seconds) to applied to the
            wrapped function. If `None`, the target function is expected to
            never timeout.
    """
    def __init__(self, timeout=..., clock=...) -> None:
        ...
    
    def __call__(self, func):
        """Apply the timeout decorator.

        Args:
            func (Callable): The function to apply the timeout argument to.
                This function must accept a timeout keyword argument.

        Returns:
            Callable: The wrapped function.
        """
        ...
    
    def __str__(self) -> str:
        ...
    


class ConstantTimeout:
    """A decorator that adds a constant timeout argument.

    DEPRECATED: use ``TimeToDeadlineTimeout`` instead.

    This is effectively equivalent to
    ``functools.partial(func, timeout=timeout)``.

    Args:
        timeout (Optional[float]): the timeout (in seconds) to applied to the
            wrapped function. If `None`, the target function is expected to
            never timeout.
    """
    def __init__(self, timeout=...) -> None:
        ...
    
    def __call__(self, func):
        """Apply the timeout decorator.

        Args:
            func (Callable): The function to apply the timeout argument to.
                This function must accept a timeout keyword argument.

        Returns:
            Callable: The wrapped function.
        """
        ...
    
    def __str__(self) -> str:
        ...
    


class ExponentialTimeout:
    """A decorator that adds an exponentially increasing timeout argument.

    DEPRECATED: the concept of incrementing timeout exponentially has been
    deprecated. Use ``TimeToDeadlineTimeout`` instead.

    This is useful if a function is called multiple times. Each time the
    function is called this decorator will calculate a new timeout parameter
    based on the the number of times the function has been called.

    For example

    .. code-block:: python

    Args:
        initial (float): The initial timeout to pass.
        maximum (float): The maximum timeout for any one call.
        multiplier (float): The multiplier applied to the timeout for each
            invocation.
        deadline (Optional[float]): The overall deadline across all
            invocations. This is used to prevent a very large calculated
            timeout from pushing the overall execution time over the deadline.
            This is especially useful in conjunction with
            :mod:`google.api_core.retry`. If ``None``, the timeouts will not
            be adjusted to accommodate an overall deadline.
    """
    def __init__(self, initial=..., maximum=..., multiplier=..., deadline=...) -> None:
        ...
    
    def with_deadline(self, deadline):
        """Return a copy of this timeout with the given deadline.

        Args:
            deadline (float): The overall deadline across all invocations.

        Returns:
            ExponentialTimeout: A new instance with the given deadline.
        """
        ...
    
    def __call__(self, func):
        """Apply the timeout decorator.

        Args:
            func (Callable): The function to apply the timeout argument to.
                This function must accept a timeout keyword argument.

        Returns:
            Callable: The wrapped function.
        """
        ...
    
    def __str__(self) -> str:
        ...
    


