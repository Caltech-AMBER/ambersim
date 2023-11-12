"""
This type stub file was generated by pyright.
"""

"""
This type stub file was generated by pyright.
"""
MAX_BODY_NAME_LENGTH = ...
Skin = ...
Bone = ...

def parse(contents, body_getter):
    """Parses the contents of a MuJoCo skin file.

    Args:
      contents: a bytes-like object containing the contents of a skin file.
      body_getter: a callable that takes a string and returns the `mjcf.Element`
        instance of a MuJoCo body of the specified name.

    Returns:
      A `Skin` named tuple.
    """
    ...

def serialize(skin):
    """Serializes a `Skin` named tuple into the contents of a MuJoCo skin file.

    Args:
      skin: a `Skin` named tuple.

    Returns:
      A `bytes` object representing the content of a MuJoCo skin file.
    """
    ...
