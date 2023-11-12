"""
This type stub file was generated by pyright.
"""

def load_path(file_obj, file_type=..., **kwargs):  # -> Path | Scene:
    """
    Load a file to a Path file_object.

    Parameters
    -----------
    file_obj : One of the following:
         - Path, Path2D, or Path3D file_objects
         - open file file_object (dxf or svg)
         - file name (dxf or svg)
         - shapely.geometry.Polygon
         - shapely.geometry.MultiLineString
         - dict with kwargs for Path constructor
         - (n,2,(2|3)) float, line segments
    file_type : str
        Type of file is required if file
        file_object passed.

    Returns
    ---------
    path : Path, Path2D, Path3D file_object
        Data as a native trimesh Path file_object
    """
    ...

def path_formats():  # -> set[str]:
    """
    Get a list of supported path formats.

    Returns
    ------------
    loaders : list of str
        Extensions of loadable formats, ie:
        ['svg', 'dxf']
    """
    ...

path_loaders = ...
