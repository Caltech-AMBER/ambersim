"""
This type stub file was generated by pyright.
"""

"""
This type stub file was generated by pyright.
"""

class NameScope:
    """A name scoping context for an MJCF model.

    This object maintains the uniqueness of identifiers within each MJCF
    namespace. Examples of MJCF namespaces include 'body', 'joint', and 'geom'.
    Each namescope also carries a name, and can have a parent namescope.
    When MJCF models are merged, all identifiers gain a hierarchical prefix
    separated by '/', which is the concatenation of all scope names up to
    the root namescope.
    """

    def __init__(self, name, mjcf_model, model_dir=..., assets=...) -> None:
        """Initializes a scope with the given name.

        Args:
          name: The scope's name
          mjcf_model: The RootElement of the MJCF model associated with this scope.
          model_dir: (optional) Path to the directory containing the model XML file.
            This is used to prefix the paths of all asset files.
          assets: (optional) A dictionary of pre-loaded assets, of the form
            `{filename: bytestring}`. If present, PyMJCF will search for assets in
            this dictionary before attempting to load them from the filesystem.
        """
        ...
    @property
    def revision(self): ...
    def increment_revision(self): ...
    @property
    def name(self):
        """This scope's name."""
        ...
    @property
    def files(self):
        """A set containing the `File` attributes registered in this scope."""
        ...
    @property
    def assets(self):
        """A dictionary containing pre-loaded assets."""
        ...
    @property
    def model_dir(self):
        """Path to the directory containing the model XML file."""
        ...
    @name.setter
    def name(self, new_name): ...
    @property
    def mjcf_model(self): ...
    @property
    def parent(self):
        """This parent `NameScope`, or `None` if this is a root scope."""
        ...
    @parent.setter
    def parent(self, new_parent): ...
    @property
    def root(self): ...
    def full_prefix(self, prefix_root=..., as_list=...):
        """The prefix for identifiers belonging to this scope.

        Args:
          prefix_root: (optional) A `NameScope` object to be treated as root
            for the purpose of calculating the prefix. If `None` then no prefix
            is produced.
          as_list: (optional) A boolean, if `True` return the list of prefix
            components. If `False`, return the full prefix string separated by
            `mjcf.constants.PREFIX_SEPARATOR`.

        Returns:
          The prefix string.
        """
        ...
    def add(self, namespace, identifier, obj):
        """Add an identifier to this name scope.

        Args:
          namespace: A string specifying the namespace to which the
            identifier belongs.
          identifier: The identifier string.
          obj: The object referred to by the identifier.

        Raises:
          ValueError: If `identifier` not valid.
        """
        ...
    def replace(self, namespace, identifier, obj):
        """Reassociates an identifier with a different object.

        Args:
          namespace: A string specifying the namespace to which the
            identifier belongs.
          identifier: The identifier string.
          obj: The object referred to by the identifier.

        Raises:
          ValueError: If `identifier` not valid.
        """
        ...
    def remove(self, namespace, identifier):
        """Removes an identifier from this name scope.

        Args:
          namespace: A string specifying the namespace to which the
            identifier belongs.
          identifier: The identifier string.

        Raises:
          KeyError: If `identifier` does not exist in this scope.
        """
        ...
    def rename(self, namespace, old_identifier, new_identifier): ...
    def get(self, namespace, identifier): ...
    def has_identifier(self, namespace, identifier): ...
