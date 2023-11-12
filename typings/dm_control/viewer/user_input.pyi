"""
This type stub file was generated by pyright.
"""

import collections

"""Utilities for handling keyboard events."""
RELEASE = ...
PRESS = ...
REPEAT = ...
KEY_UNKNOWN = ...
KEY_SPACE = ...
KEY_APOSTROPHE = ...
KEY_COMMA = ...
KEY_MINUS = ...
KEY_PERIOD = ...
KEY_SLASH = ...
KEY_0 = ...
KEY_1 = ...
KEY_2 = ...
KEY_3 = ...
KEY_4 = ...
KEY_5 = ...
KEY_6 = ...
KEY_7 = ...
KEY_8 = ...
KEY_9 = ...
KEY_SEMICOLON = ...
KEY_EQUAL = ...
KEY_A = ...
KEY_B = ...
KEY_C = ...
KEY_D = ...
KEY_E = ...
KEY_F = ...
KEY_G = ...
KEY_H = ...
KEY_I = ...
KEY_J = ...
KEY_K = ...
KEY_L = ...
KEY_M = ...
KEY_N = ...
KEY_O = ...
KEY_P = ...
KEY_Q = ...
KEY_R = ...
KEY_S = ...
KEY_T = ...
KEY_U = ...
KEY_V = ...
KEY_W = ...
KEY_X = ...
KEY_Y = ...
KEY_Z = ...
KEY_LEFT_BRACKET = ...
KEY_BACKSLASH = ...
KEY_RIGHT_BRACKET = ...
KEY_GRAVE_ACCENT = ...
KEY_ESCAPE = ...
KEY_ENTER = ...
KEY_TAB = ...
KEY_BACKSPACE = ...
KEY_INSERT = ...
KEY_DELETE = ...
KEY_RIGHT = ...
KEY_LEFT = ...
KEY_DOWN = ...
KEY_UP = ...
KEY_PAGE_UP = ...
KEY_PAGE_DOWN = ...
KEY_HOME = ...
KEY_END = ...
KEY_CAPS_LOCK = ...
KEY_SCROLL_LOCK = ...
KEY_NUM_LOCK = ...
KEY_PRINT_SCREEN = ...
KEY_PAUSE = ...
KEY_F1 = ...
KEY_F2 = ...
KEY_F3 = ...
KEY_F4 = ...
KEY_F5 = ...
KEY_F6 = ...
KEY_F7 = ...
KEY_F8 = ...
KEY_F9 = ...
KEY_F10 = ...
KEY_F11 = ...
KEY_F12 = ...
KEY_KP_0 = ...
KEY_KP_1 = ...
KEY_KP_2 = ...
KEY_KP_3 = ...
KEY_KP_4 = ...
KEY_KP_5 = ...
KEY_KP_6 = ...
KEY_KP_7 = ...
KEY_KP_8 = ...
KEY_KP_9 = ...
KEY_KP_DECIMAL = ...
KEY_KP_DIVIDE = ...
KEY_KP_MULTIPLY = ...
KEY_KP_SUBTRACT = ...
KEY_KP_ADD = ...
KEY_KP_ENTER = ...
KEY_KP_EQUAL = ...
KEY_LEFT_SHIFT = ...
KEY_LEFT_CONTROL = ...
KEY_LEFT_ALT = ...
KEY_LEFT_SUPER = ...
KEY_RIGHT_SHIFT = ...
KEY_RIGHT_CONTROL = ...
KEY_RIGHT_ALT = ...
KEY_RIGHT_SUPER = ...
MOD_NONE = ...
MOD_SHIFT = ...
MOD_CONTROL = ...
MOD_ALT = ...
MOD_SUPER = ...
MOD_SHIFT_CONTROL = ...
MOUSE_BUTTON_LEFT = ...
MOUSE_BUTTON_RIGHT = ...
MOUSE_BUTTON_MIDDLE = ...
_NO_EXCLUSIVE_KEY = ...
_NO_CALLBACK = ...

class Exclusive(collections.namedtuple("Exclusive", "combination")):
    """Defines an exclusive action.

    Exclusive actions can be invoked in response to single key clicks only. The
    callback will be called twice. The first time when the key combination is
    pressed, passing True as the argument to the callback. The second time when
    the key is released (the modifiers don't have to be present then), passing
    False as the callback argument.

    Attributes:
      combination: A list of integers interpreted as key codes, or tuples
        in format (keycode, modifier).
    """

    ...

class DoubleClick(collections.namedtuple("DoubleClick", "combination")):
    """Defines a mouse double click action.

    It will define a requirement to double click the mouse button specified in the
    combination in order to be triggered.

    Attributes:
      combination: A list of integers interpreted as key codes, or tuples
        in format (keycode, modifier). The keycodes are limited only to mouse
        button codes.
    """

    ...

class Range(collections.namedtuple("Range", "collection")):
    """Binds a number of key combinations to a callback.

    When triggered, the index of the triggering key combination will be passed
    as an argument to the callback.

    Attributes:
      callback: A callable accepting a single argument - an integer index of the
        triggered callback.
      collection: A collection of combinations. Combinations may either be raw key
        codes, tuples in format (keycode, modifier), or one of the Exclusive or
        DoubleClick instances.
    """

    ...

class InputMap:
    """Provides ability to alias key combinations and map them to actions."""

    def __init__(self, mouse, keyboard) -> None:
        """Instance initializer.

        Args:
          mouse: GlfwMouse instance.
          keyboard: GlfwKeyboard instance.
        """
        ...
    def __del__(self):  # -> None:
        """Instance deleter."""
        ...
    def clear_bindings(self):  # -> None:
        """Clears registered action bindings, while keeping key aliases."""
        ...
    def bind(self, callback, key_binding):  # -> None:
        """Binds a key combination to a callback.

        Args:
          callback: An argument-less callable.
          key_binding: A integer with a key code, a tuple (keycode, modifier) or one
            of the actions Exclusive|DoubleClick|Range carrying the key combination.
        """
        ...
    def bind_plane(self, callback):  # -> None:
        """Binds a callback to a planar motion action (mouse movement)."""
        ...
    def bind_z_axis(self, callback):  # -> None:
        """Binds a callback to a z-axis motion action (mouse scroll)."""
        ...
