"""
This type stub file was generated by pyright.
"""

from collections.abc import Mapping

import numpy as np

"""
caching.py
-----------

Functions and classes that help with tracking changes
in `numpy.ndarray` and clearing cached values based
on those changes.

You should really `pip install xxhash`:

```
In [23]: %timeit int(blake2b(d).hexdigest(), 16)
102 us +/- 684 ns per loop

In [24]: %timeit int(sha256(d).hexdigest(), 16)
142 us +/- 3.73 us

In [25]: %timeit xxh3_64_intdigest(d)
3.37 us +/- 116 ns per loop
```
"""

def sha256(item):  # -> int:
    ...

def hash_fallback(item):  # -> int:
    ...

def tracked_array(array, dtype=...):  # -> TrackedArray:
    """
    Properly subclass a numpy ndarray to track changes.

    Avoids some pitfalls of subclassing by forcing contiguous
    arrays and does a view into a TrackedArray.

    Parameters
    ------------
    array : array- like object
      To be turned into a TrackedArray
    dtype : np.dtype
      Which dtype to use for the array

    Returns
    ------------
    tracked : TrackedArray
      Contains input array data.
    """
    ...

def cache_decorator(function):  # -> property:
    """
    A decorator for class methods, replaces @property
    but will store and retrieve function return values
    in object cache.

    Parameters
    ------------
    function : method
      This is used as a decorator:
      ```
      @cache_decorator
      def foo(self, things):
        return 'happy days'
      ```
    """
    ...

class TrackedArray(np.ndarray):
    """
    Subclass of numpy.ndarray that provides hash methods
    to track changes.

    General method is to aggressively set 'modified' flags
    on operations which might (but don't necessarily) alter
    the array, ideally we sometimes compute hashes when we
    don't need to, but we don't return wrong hashes ever.

    We store boolean modified flag for each hash type to
    make checks fast even for queries of different hashes.

    Methods
    ----------
    __hash__ : int
      Runs the fastest available hash in this order:
        `xxh3_64, xxh_64, blake2b, sha256`
    """

    def __array_finalize__(self, obj):  # -> None:
        """
        Sets a modified flag on every TrackedArray
        This flag will be set on every change as well as
        during copies and certain types of slicing.
        """
        ...
    def __array_wrap__(self, out_arr, context=...):  # -> ndarray[Unknown, Unknown]:
        """
        Return a numpy scalar if array is 0d.
        See https://github.com/numpy/numpy/issues/5819
        """
        ...
    @property
    def mutable(self):  # -> bool:
        ...
    @mutable.setter
    def mutable(self, value):  # -> None:
        ...
    def __hash__(self) -> int:
        """
        Return a fast hash of the contents of the array.

        Returns
        -------------
        hash : long int
          A hash of the array contents.
        """
        ...
    def __iadd__(self, *args, **kwargs):
        """
        In-place addition.

        The i* operations are in- place and modify the array,
        so we better catch all of them.
        """
        ...
    def __isub__(self, *args, **kwargs): ...
    def fill(self, *args, **kwargs):  # -> None:
        ...
    def partition(self, *args, **kwargs):  # -> None:
        ...
    def put(self, *args, **kwargs):  # -> None:
        ...
    def byteswap(self, *args, **kwargs):  # -> Self@TrackedArray:
        ...
    def itemset(self, *args, **kwargs):  # -> None:
        ...
    def sort(self, *args, **kwargs):  # -> None:
        ...
    def setflags(self, *args, **kwargs):  # -> None:
        ...
    def __imul__(self, *args, **kwargs): ...
    def __idiv__(self, *args, **kwargs): ...
    def __itruediv__(self, *args, **kwargs): ...
    def __imatmul__(self, *args, **kwargs): ...
    def __ipow__(self, *args, **kwargs): ...
    def __imod__(self, *args, **kwargs): ...
    def __ifloordiv__(self, *args, **kwargs): ...
    def __ilshift__(self, *args, **kwargs): ...
    def __irshift__(self, *args, **kwargs): ...
    def __iand__(self, *args, **kwargs): ...
    def __ixor__(self, *args, **kwargs): ...
    def __ior__(self, *args, **kwargs): ...
    def __setitem__(self, *args, **kwargs): ...
    def __setslice__(self, *args, **kwargs): ...

class Cache:
    """
    Class to cache values which will be stored until the
    result of an ID function changes.
    """

    def __init__(self, id_function, force_immutable=...) -> None:
        """
        Create a cache object.

        Parameters
        ------------
        id_function : function
          Returns hashable value
        force_immutable : bool
          If set will make all numpy arrays read-only
        """
        ...
    def delete(self, key):  # -> None:
        """
        Remove a key from the cache.
        """
        ...
    def verify(self):  # -> None:
        """
        Verify that the cached values are still for the same
        value of id_function and delete all stored items if
        the value of id_function has changed.
        """
        ...
    def clear(self, exclude=...):  # -> None:
        """
        Remove elements in the cache.

        Parameters
        -----------
        exclude : list
          List of keys in cache to not clear.
        """
        ...
    def update(self, items):  # -> None:
        """
        Update the cache with a set of key, value pairs without
        checking id_function.
        """
        ...
    def id_set(self):  # -> None:
        """
        Set the current ID to the value of the ID function.
        """
        ...
    def __getitem__(self, key):  # -> None:
        """
        Get an item from the cache. If the item
        is not in the cache, it will return None

        Parameters
        -------------
        key : hashable
               Key in dict

        Returns
        -------------
        cached : object, or None
          Object that was stored
        """
        ...
    def __setitem__(self, key, value):
        """
        Add an item to the cache.

        Parameters
        ------------
        key : hashable
          Key to reference value
        value : any
          Value to store in cache
        """
        ...
    def __contains__(self, key):  # -> bool:
        ...
    def __len__(self):  # -> int:
        ...
    def __enter__(self):  # -> None:
        ...
    def __exit__(self, *args):  # -> None:
        ...

class DiskCache:
    """
    Store results of expensive operations on disk
    with an option to expire the results. This is used
    to cache the multi-gigabyte test corpuses in
    `tests/corpus.py`
    """

    def __init__(self, path, expire_days=...) -> None:
        """
        Create a cache on disk for storing expensive results.

        Parameters
        --------------
        path : str
          A writeable location on the current file path.
        expire_days : int or float
          How old should results be considered expired.

        """
        ...
    def get(self, key, fetch):  # -> bytes:
        """
        Get a key from the cache or run a calculation.

        Parameters
        -----------
        key : str
          Key to reference item with
        fetch : function
          If key isn't stored and recent run this
          function and store its result on disk.
        """
        ...

class DataStore(Mapping):
    """
    A class to store multiple numpy arrays and track them all
    for changes.

    Operates like a dict that only stores numpy.ndarray
    """

    def __init__(self) -> None: ...
    def __iter__(self):  # -> Iterator[Unknown]:
        ...
    def pop(self, key): ...
    def __delitem__(self, key):  # -> None:
        ...
    @property
    def mutable(self):  # -> Any | bool:
        """
        Is data allowed to be altered or not.

        Returns
        -----------
        is_mutable : bool
          Can data be altered in the DataStore
        """
        ...
    @mutable.setter
    def mutable(self, value):  # -> None:
        """
        Is data allowed to be altered or not.

        Parameters
        ------------
        is_mutable : bool
          Should data be allowed to be altered
        """
        ...
    def is_empty(self):  # -> bool:
        """
        Is the current DataStore empty or not.

        Returns
        ----------
        empty : bool
          False if there are items in the DataStore
        """
        ...
    def clear(self):  # -> None:
        """
        Remove all data from the DataStore.
        """
        ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, data):  # -> None:
        """
        Store an item in the DataStore.

        Parameters
        -------------
        key
          A hashable key to store under
        data
          Usually a numpy array which will be subclassed
          but anything hashable should be able to be stored.
        """
        ...
    def __contains__(self, key):  # -> bool:
        ...
    def __len__(self):  # -> int:
        ...
    def update(self, values):  # -> None:
        ...
    def __hash__(self) -> int:
        """
        Get a hash reflecting everything in the DataStore.

        Returns
        ----------
        hash : str
          hash of data in hexadecimal
        """
        ...
