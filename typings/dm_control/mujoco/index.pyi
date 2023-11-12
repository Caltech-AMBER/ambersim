"""
This type stub file was generated by pyright.
"""

import abc

"""Mujoco functions to support named indexing.

The Mujoco name structure works as follows:

In mjxmacro.h, each "X" entry denotes a type (a), a field name (b) and a list
of dimension size metadata (c) which may contain both numbers and names, for
example

   X(int,    name_bodyadr, nbody, 1) // or
   X(mjtNum, body_pos,     nbody, 3)
     a       b             c ----->

The second declaration states that the field `body_pos` has type `mjtNum` and
dimension sizes `(nbody, 3)`, i.e. the first axis is indexed by body number.
These and other named dimensions are sized based on the loaded model. This
information is parsed and stored in `mjbindings.sizes`.

In mjmodel.h, the struct mjModel contains an array of element name addresses
for each size name.

   int* name_bodyadr; // body name pointers (nbody x 1)

By iterating over each of these element name address arrays, we first obtain a
mapping from size names to a list of element names.

    {'nbody': ['cart', 'pole'], 'njnt': ['free', 'ball', 'hinge'], ...}

In addition to the element names that are derived from the mjModel struct at
runtime, we also assign hard-coded names to certain dimensions where there is an
established naming convention (e.g. 'x', 'y', 'z' for dimensions that correspond
to Cartesian positions).

For some dimensions, a single element name maps to multiple indices within the
underlying field. For example, a single joint name corresponds to a variable
number of indices within `qpos` that depends on the number of degrees of freedom
associated with that joint type. These are referred to as "ragged" dimensions.

In such cases we determine the size of each named element by examining the
address arrays (e.g. `jnt_qposadr`), and construct a mapping from size name to
element sizes:

    {'nq': [7, 3, 1], 'nv': [6, 3, 1], ...}

Given these two dictionaries, we then create an `Axis` instance for each size
name. These objects have a `convert_key_item` method that handles the conversion
from indexing expressions containing element names to valid numpy indices.
Different implementations of `Axis` are used to handle "ragged" and "non-ragged"
dimensions.

    {'nbody': RegularNamedAxis(names=['cart', 'pole']),
     'nq': RaggedNamedAxis(names=['free', 'ball', 'hinge'], sizes=[7, 4, 1])}

We construct this dictionary once using `make_axis_indexers`.

Finally, for each field we construct a `FieldIndexer` class. A `FieldIndexer`
instance encapsulates a field together with a list of `Axis` instances (one per
dimension), and implements the named indexing logic by calling their respective
`convert_key_item` methods.

Summary of terminology:

* _size name_ or _size_ A dimension size name, e.g. `nbody` or `ngeom`.
* _element name_ or _name_ A named element in a Mujoco model, e.g. 'cart' or
  'pole'.
* _element index_ or _index_ The index of an element name, for a specific size
  name.
"""
_RAGGED_ADDRS = ...
_COLUMN_NAMES = ...
_COLUMN_ID_TO_FIELDS = ...

def make_axis_indexers(model):  # -> defaultdict[Unknown, UnnamedAxis]:
    """Returns a dict that maps size names to `Axis` indexers.

    Args:
      model: An instance of `mjbindings.MjModelWrapper`.

    Returns:
      A `dict` mapping from a size name (e.g. `'nbody'`) to an `Axis` instance.
    """
    ...

class Axis(metaclass=abc.ABCMeta):
    """Handles the conversion of named indexing expressions into numpy indices."""

    @abc.abstractmethod
    def convert_key_item(self, key_item):  # -> None:
        """Converts a (possibly named) indexing expression to a numpy index."""
        ...

class UnnamedAxis(Axis):
    """An object representing an axis where the elements are not named."""

    def convert_key_item(self, key_item):
        """Validate the indexing expression and return it unmodified."""
        ...

class RegularNamedAxis(Axis):
    """Represents an axis where each named element has a fixed size of 1."""

    def __init__(self, names) -> None:
        """Initializes a new `RegularNamedAxis` instance.

        Args:
          names: A list or array of element names.
        """
        ...
    def convert_key_item(self, key_item):  # -> int | NDArray[Any] | NDArray[Unknown]:
        """Converts a named indexing expression to a numpy-friendly index."""
        ...
    @property
    def names(self):  # -> Unknown:
        """Returns a list of element names."""
        ...

class RaggedNamedAxis(Axis):
    """Represents an axis where the named elements may vary in size."""

    def __init__(self, element_names, element_sizes, singleton=...) -> None:
        """Initializes a new `RaggedNamedAxis` instance.

        Args:
          element_names: A list or array containing the element names.
          element_sizes: A list or array containing the size of each element.
          singleton: Whether to reduce singleton slices to scalars.
        """
        ...
    def convert_key_item(self, key_item):  # -> list[Unknown] | ndarray[Unknown, Unknown]:
        """Converts a named indexing expression to a numpy-friendly index."""
        ...
    @property
    def names(self):  # -> Unknown:
        """Returns a list of element names."""
        ...

Axes = ...

class FieldIndexer:
    """An array-like object providing named access to a field in a MuJoCo struct.

    FieldIndexers expose the same attributes and methods as an `np.ndarray`.

    They may be indexed with strings or lists of strings corresponding to element
    names. They also support standard numpy indexing expressions, with the
    exception of indices containing `Ellipsis` or `None`.
    """

    __slots__ = ...
    def __init__(self, parent_struct, field_name, axis_indexers) -> None:
        """Initializes a new `FieldIndexer`.

        Args:
          parent_struct: Wrapped ctypes structure, as generated by `mjbindings`.
          field_name: String containing field name in `parent_struct`.
          axis_indexers: A list of `Axis` instances, one per dimension.
        """
        ...
    def __dir__(self):  # -> list[str]:
        ...
    def __getattr__(self, name):  # -> Any:
        ...
    def __getitem__(self, key):
        """Converts the key to a numeric index and returns the indexed array.

        Args:
          key: Indexing expression.

        Raises:
          IndexError: If an indexing tuple has too many elements, or if it contains
            `Ellipsis`, `None`, or an empty string.

        Returns:
          The indexed array.
        """
        ...
    def __setitem__(self, key, value):  # -> None:
        """Converts the key and assigns to the indexed array.

        Args:
          key: Indexing expression.
          value: Value to assign.

        Raises:
          IndexError: If an indexing tuple has too many elements, or if it contains
            `Ellipsis`, `None`, or an empty string.
        """
        ...
    @property
    def axes(self):  # -> Axes:
        """A namedtuple containing the row and column indexers for this field."""
        ...
    def __repr__(self):  # -> LiteralString:
        """Returns a pretty string representation of the `FieldIndexer`."""
        ...

def struct_indexer(struct, struct_name, size_to_axis_indexer):  # -> StructIndexer:
    """Returns an object with a `FieldIndexer` attribute for each dynamic field.

    Usage example

    ```python
    named_data = struct_indexer(mjdata, 'mjdata', size_to_axis_indexer)
    fingertip_xpos = named_data.xpos['fingertip']
    elbow_qvel = named_data.qvel['elbow']
    ```

    Args:
      struct: Wrapped ctypes structure as generated by `mjbindings`.
      struct_name: String containing corresponding Mujoco name of struct.
      size_to_axis_indexer: dict that maps size names to `Axis` instances.

    Returns:
      An object with a field for every dynamically sized array field, mapping to a
      `FieldIndexer`. The returned object is immutable and has an `_asdict`
      method.

    Raises:
      ValueError: If `struct_name` is not recognized.
    """
    ...

def make_struct_indexer(field_indexers):  # -> StructIndexer:
    """Returns an immutable container exposing named indexers as attributes."""

    class StructIndexer: ...
