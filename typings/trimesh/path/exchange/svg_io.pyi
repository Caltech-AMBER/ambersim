"""
This type stub file was generated by pyright.
"""

import numpy as np

_ns_name = ...
_ns_url = ...
_ns = ...
_IDENTITY = np.eye(3)

def svg_to_path(
    file_obj=..., file_type=..., path_string=...
):  # -> dict[str, list[Unknown]] | dict[str, dict[Unknown, dict[str, Unknown]]] | dict[str, Unknown]:
    """
    Load an SVG file into a Path2D object.

    Parameters
    -----------
    file_obj : open file object
      Contains SVG data
    file_type: None
      Not used
    path_string : None or str
      If passed, parse a single path string and ignore `file_obj`.

    Returns
    -----------
    loaded : dict
      With kwargs for Path2D constructor
    """
    ...

def transform_to_matrices(transform):  # -> list[Unknown]:
    """
    Convert an SVG transform string to an array of matrices.

    i.e. "rotate(-10 50 100)
          translate(-36 45.5)
          skewX(40)
          scale(1 0.5)"

    Parameters
    -----------
    transform : str
      Contains transformation information in SVG form

    Returns
    -----------
    matrices : (n, 3, 3) float
      Multiple transformation matrices from input transform string
    """
    ...

def export_svg(drawing, return_path=..., only_layers=..., digits=..., **kwargs):  # -> LiteralString | str | Any:
    """
    Export a Path2D object into an SVG file.

    Parameters
    -----------
    drawing : Path2D
     Source geometry
    return_path : bool
      If True return only path string not wrapped in XML
    only_layers : None or set
      If passed only export the specified layers
    digits : None or int
      Number of digits for floating point values

    Returns
    -----------
    as_svg : str
      XML formatted SVG, or path string
    """
    ...
