"""
This type stub file was generated by pyright.
"""

import datetime

"""
This type stub file was generated by pyright.
"""
_UTC_EPOCH = ...
_RFC3339_MICROS = ...
_RFC3339_NO_FRACTION = ...
_RFC3339_NANOS = ...
def utcnow():
    """A :meth:`datetime.datetime.utcnow()` alias to allow mocking in tests."""
    ...

def to_milliseconds(value):
    """Convert a zone-aware datetime to milliseconds since the unix epoch.

    Args:
        value (datetime.datetime): The datetime to covert.

    Returns:
        int: Milliseconds since the unix epoch.
    """
    ...

def from_microseconds(value):
    """Convert timestamp in microseconds since the unix epoch to datetime.

    Args:
        value (float): The timestamp to convert, in microseconds.

    Returns:
        datetime.datetime: The datetime object equivalent to the timestamp in
            UTC.
    """
    ...

def to_microseconds(value):
    """Convert a datetime to microseconds since the unix epoch.

    Args:
        value (datetime.datetime): The datetime to covert.

    Returns:
        int: Microseconds since the unix epoch.
    """
    ...

def from_iso8601_date(value):
    """Convert a ISO8601 date string to a date.

    Args:
        value (str): The ISO8601 date string.

    Returns:
        datetime.date: A date equivalent to the date string.
    """
    ...

def from_iso8601_time(value):
    """Convert a zoneless ISO8601 time string to a time.

    Args:
        value (str): The ISO8601 time string.

    Returns:
        datetime.time: A time equivalent to the time string.
    """
    ...

def from_rfc3339(value):
    """Convert an RFC3339-format timestamp to a native datetime.

    Supported formats include those without fractional seconds, or with
    any fraction up to nanosecond precision.

    .. note::
        Python datetimes do not support nanosecond precision; this function
        therefore truncates such values to microseconds.

    Args:
        value (str): The RFC3339 string to convert.

    Returns:
        datetime.datetime: The datetime object equivalent to the timestamp
        in UTC.

    Raises:
        ValueError: If the timestamp does not match the RFC3339
            regular expression.
    """
    ...

from_rfc3339_nanos = ...
def to_rfc3339(value, ignore_zone=...):
    """Convert a datetime to an RFC3339 timestamp string.

    Args:
        value (datetime.datetime):
            The datetime object to be converted to a string.
        ignore_zone (bool): If True, then the timezone (if any) of the
            datetime object is ignored and the datetime is treated as UTC.

    Returns:
        str: The RFC3339 formatted string representing the datetime.
    """
    ...

class DatetimeWithNanoseconds(datetime.datetime):
    """Track nanosecond in addition to normal datetime attrs.

    Nanosecond can be passed only as a keyword argument.
    """
    __slots__ = ...
    def __new__(cls, *args, **kw):
        ...
    
    @property
    def nanosecond(self):
        """Read-only: nanosecond precision."""
        ...
    
    def rfc3339(self):
        """Return an RFC3339-compliant timestamp.

        Returns:
            (str): Timestamp string according to RFC3339 spec.
        """
        ...
    
    @classmethod
    def from_rfc3339(cls, stamp):
        """Parse RFC3339-compliant timestamp, preserving nanoseconds.

        Args:
            stamp (str): RFC3339 stamp, with up to nanosecond precision

        Returns:
            :class:`DatetimeWithNanoseconds`:
                an instance matching the timestamp string

        Raises:
            ValueError: if `stamp` does not match the expected format
        """
        ...
    
    def timestamp_pb(self):
        """Return a timestamp message.

        Returns:
            (:class:`~google.protobuf.timestamp_pb2.Timestamp`): Timestamp message
        """
        ...
    
    @classmethod
    def from_timestamp_pb(cls, stamp):
        """Parse RFC3339-compliant timestamp, preserving nanoseconds.

        Args:
            stamp (:class:`~google.protobuf.timestamp_pb2.Timestamp`): timestamp message

        Returns:
            :class:`DatetimeWithNanoseconds`:
                an instance matching the timestamp message
        """
        ...
    


