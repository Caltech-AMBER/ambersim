"""
This type stub file was generated by pyright.
"""

import json
import os

from ..util import decode_text, wrap_as_stream

_pwd = ...
_cache = ...

def get(name, decode=..., decode_json=..., as_stream=...):  # -> Any | StringIO | BytesIO | bytes | str:
    """
    Get a resource from the `trimesh/resources` folder.

    Parameters
    -------------
    name : str
      File path relative to `trimesh/resources`
    decode : bool
      Whether or not to decode result as UTF-8
    decode_json : bool
      Run `json.loads` on resource if True.
    as_stream : bool
      Return as a file-like object

    Returns
    -------------
    resource : str, bytes, or decoded JSON
      File data
    """
    ...

def get_schema(name):  # -> list[Unknown] | dict[Unknown, Unknown] | Any:
    """
    Load a schema and evaluate the referenced files.

    Parameters
    ------------
    name : str
      Filename of schema.

    Returns
    ----------
    schema : dict
      Loaded and resolved schema.
    """
    ...
