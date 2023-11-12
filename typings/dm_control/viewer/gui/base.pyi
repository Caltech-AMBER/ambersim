"""
This type stub file was generated by pyright.
"""

import abc

"""Utilities and base classes used exclusively in the gui package."""
_DOUBLE_CLICK_INTERVAL = ...

class InputEventsProcessor(metaclass=abc.ABCMeta):
    """Thread safe input events processor."""

    def __init__(self) -> None:
        """Instance initializer."""
        ...
    def add_event(self, receivers, *args):  # -> None:
        """Adds a new event to the processing queue."""
        ...
    def process_events(self):  # -> None:
        """Invokes each of the events in the queue.

        Thread safe for queue access but not during event invocations.

        This method must be called regularly on the main thread.
        """
        ...

class DoubleClickDetector:
    """Detects double click events."""

    def __init__(self) -> None: ...
    def process(self, button, action):  # -> bool:
        """Attempts to identify a mouse button click as a double click event."""
        ...
