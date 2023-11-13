"""
This type stub file was generated by pyright.
"""

class ContactData:
    """
    Data structure for holding information about a collision contact.
    """

    def __init__(self, names, contact) -> None:
        """
        Initialize a ContactData.

        Parameters
        ----------
        names : list of str
          The names of the two objects in order.
        contact : fcl.Contact
          The contact in question.
        """
        ...
    @property
    def normal(self):
        """
        The 3D intersection normal for this contact.

        Returns
        -------
        normal : (3,) float
          The intersection normal.
        """
        ...
    @property
    def point(self):
        """
        The 3D point of intersection for this contact.

        Returns
        -------
        point : (3,) float
          The intersection point.
        """
        ...
    @property
    def depth(self):
        """
        The penetration depth of the 3D point of intersection for this contact.

        Returns
        -------
        depth : float
          The penetration depth.
        """
        ...
    def index(self, name):
        """
        Returns the index of the face in contact for the mesh with
        the given name.

        Parameters
        ----------
        name : str
          The name of the target object.

        Returns
        -------
        index : int
          The index of the face in collision
        """
        ...

class DistanceData:
    """
    Data structure for holding information about a distance query.
    """

    def __init__(self, names, result) -> None:
        """
        Initialize a DistanceData.

        Parameters
        ----------
        names : list of str
          The names of the two objects in order.
        contact : fcl.DistanceResult
          The distance query result.
        """
        ...
    @property
    def distance(self):
        """
        Returns the distance between the two objects.

        Returns
        -------
        distance : float
          The euclidean distance between the objects.
        """
        ...
    def index(self, name):
        """
        Returns the index of the closest face for the mesh with
        the given name.

        Parameters
        ----------
        name : str
          The name of the target object.

        Returns
        -------
        index : int
          The index of the face in collisoin.
        """
        ...
    def point(self, name):
        """
        The 3D point of closest distance on the mesh with the given name.

        Parameters
        ----------
        name : str
          The name of the target object.

        Returns
        -------
        point : (3,) float
          The closest point.
        """
        ...

class CollisionManager:
    """
    A mesh-mesh collision manager.
    """

    def __init__(self) -> None:
        """
        Initialize a mesh-mesh collision manager.
        """
        ...
    def add_object(self, name, mesh, transform=...):
        """
        Add an object to the collision manager.

        If an object with the given name is already in the manager,
        replace it.

        Parameters
        ----------
        name : str
          An identifier for the object
        mesh : Trimesh object
          The geometry of the collision object
        transform : (4,4) float
          Homogeneous transform matrix for the object
        """
        ...
    def remove_object(self, name):  # -> None:
        """
        Delete an object from the collision manager.

        Parameters
        ----------
        name : str
          The identifier for the object
        """
        ...
    def set_transform(self, name, transform):  # -> None:
        """
        Set the transform for one of the manager's objects.
        This replaces the prior transform.

        Parameters
        ----------
        name : str
          An identifier for the object already in the manager
        transform : (4,4) float
          A new homogeneous transform matrix for the object
        """
        ...
    def in_collision_single(
        self, mesh, transform=..., return_names=..., return_data=...
    ):  # -> tuple[Unknown, set[Unknown], list[Unknown]] | tuple[Unknown, set[Unknown]] | tuple[Unknown, list[Unknown]]:
        """
        Check a single object for collisions against all objects in the
        manager.

        Parameters
        ----------
        mesh : Trimesh object
          The geometry of the collision object
        transform : (4,4) float
          Homogeneous transform matrix
        return_names : bool
          If true, a set is returned containing the names
          of all objects in collision with the object
        return_data :  bool
          If true, a list of ContactData is returned as well

        Returns
        ------------
        is_collision : bool
          True if a collision occurs and False otherwise
        names : set of str
          [OPTIONAL] The set of names of objects that collided with the
          provided one
        contacts : list of ContactData
          [OPTIONAL] All contacts detected
        """
        ...
    def in_collision_internal(
        self, return_names=..., return_data=...
    ):  # -> tuple[Unknown, set[Unknown], list[Unknown]] | tuple[Unknown, set[Unknown]] | tuple[Unknown, list[Unknown]]:
        """
        Check if any pair of objects in the manager collide with one another.

        Parameters
        ----------
        return_names : bool
          If true, a set is returned containing the names
          of all pairs of objects in collision.
        return_data :  bool
          If true, a list of ContactData is returned as well

        Returns
        -------
        is_collision : bool
          True if a collision occurred between any pair of objects
          and False otherwise
        names : set of 2-tup
          The set of pairwise collisions. Each tuple
          contains two names in alphabetical order indicating
          that the two corresponding objects are in collision.
        contacts : list of ContactData
          All contacts detected
        """
        ...
    def in_collision_other(
        self, other_manager, return_names=..., return_data=...
    ):  # -> tuple[Unknown, set[Unknown], list[Unknown]] | tuple[Unknown, set[Unknown]] | tuple[Unknown, list[Unknown]]:
        """
        Check if any object from this manager collides with any object
        from another manager.

        Parameters
        -------------------
        other_manager : CollisionManager
          Another collision manager object
        return_names : bool
          If true, a set is returned containing the names
          of all pairs of objects in collision.
        return_data : bool
          If true, a list of ContactData is returned as well

        Returns
        -------------
        is_collision : bool
          True if a collision occurred between any pair of objects
          and False otherwise
        names : set of 2-tup
          The set of pairwise collisions. Each tuple
          contains two names (first from this manager,
          second from the other_manager) indicating
          that the two corresponding objects are in collision.
        contacts : list of ContactData
          All contacts detected
        """
        ...
    def min_distance_single(
        self, mesh, transform=..., return_name=..., return_data=...
    ):  # -> tuple[Unknown, None, DistanceData | None] | tuple[Unknown, None] | tuple[Unknown, DistanceData | None]:
        """
        Get the minimum distance between a single object and any
        object in the manager.

        Parameters
        ---------------
        mesh : Trimesh object
          The geometry of the collision object
        transform : (4,4) float
          Homogeneous transform matrix for the object
        return_names : bool
          If true, return name of the closest object
        return_data : bool
          If true, a DistanceData object is returned as well

        Returns
        -------------
        distance : float
          Min distance between mesh and any object in the manager
        name : str
          The name of the object in the manager that was closest
        data : DistanceData
          Extra data about the distance query
        """
        ...
    def min_distance_internal(
        self, return_names=..., return_data=...
    ):  # -> tuple[Unknown, tuple[Unknown, ...] | None, DistanceData | None] | tuple[Unknown, tuple[Unknown, ...] | None] | tuple[Unknown, DistanceData | None]:
        """
        Get the minimum distance between any pair of objects in the manager.

        Parameters
        -------------
        return_names : bool
          If true, a 2-tuple is returned containing the names
          of the closest objects.
        return_data : bool
          If true, a DistanceData object is returned as well

        Returns
        -----------
        distance : float
          Min distance between any two managed objects
        names : (2,) str
          The names of the closest objects
        data : DistanceData
          Extra data about the distance query
        """
        ...
    def min_distance_other(
        self, other_manager, return_names=..., return_data=...
    ):  # -> tuple[Unknown, tuple[None, Unknown] | None, DistanceData | None] | tuple[Unknown, tuple[None, Unknown] | None] | tuple[Unknown, DistanceData | None]:
        """
        Get the minimum distance between any pair of objects,
        one in each manager.

        Parameters
        ----------
        other_manager : CollisionManager
          Another collision manager object
        return_names : bool
          If true, a 2-tuple is returned containing
          the names of the closest objects.
        return_data : bool
          If true, a DistanceData object is returned as well

        Returns
        -----------
        distance : float
          The min distance between a pair of objects,
          one from each manager.
        names : 2-tup of str
          A 2-tuple containing two names (first from this manager,
          second from the other_manager) indicating
          the two closest objects.
        data : DistanceData
          Extra data about the distance query
        """
        ...

def mesh_to_BVH(mesh):
    """
    Create a BVHModel object from a Trimesh object

    Parameters
    -----------
    mesh : Trimesh
      Input geometry

    Returns
    ------------
    bvh : fcl.BVHModel
      BVH of input geometry
    """
    ...

def mesh_to_convex(mesh):
    """
    Create a Convex object from a Trimesh object

    Parameters
    -----------
    mesh : Trimesh
      Input geometry

    Returns
    ------------
    convex : fcl.Convex
      Convex of input geometry
    """
    ...

def scene_to_collision(scene):  # -> tuple[CollisionManager, dict[Unknown, Unknown]]:
    """
    Create collision objects from a trimesh.Scene object.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to create collision objects for

    Returns
    ------------
    manager : CollisionManager
      CollisionManager for objects in scene
    objects: {node name: CollisionObject}
      Collision objects for nodes in scene
    """
    ...
