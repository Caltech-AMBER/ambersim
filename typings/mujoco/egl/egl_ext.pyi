"""
This type stub file was generated by pyright.
"""

from OpenGL.EGL import *

"""Extends OpenGL.EGL with definitions necessary for headless rendering."""
PFNEGLQUERYDEVICESEXTPROC = ...
_eglQueryDevicesEXT = ...
EGL_PLATFORM_DEVICE_EXT = ...
PFNEGLGETPLATFORMDISPLAYEXTPROC = ...
eglGetPlatformDisplayEXT = ...

def eglQueryDevicesEXT(max_devices=...):  # -> list[Any]:
    ...
