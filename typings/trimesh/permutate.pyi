"""
This type stub file was generated by pyright.
"""

"""
permutate.py
-------------

Randomly deform meshes in different ways.
"""

def transform(mesh, translation_scale=...):  # -> Any:
    """
    Return a permutated variant of a mesh by randomly reording faces
    and rotatating + translating a mesh by a random matrix.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh, will not be altered by this function

    Returns
    ----------
    permutated : trimesh.Trimesh
      Mesh with same faces as input mesh but reordered
      and rigidly transformed in space.
    """
    ...

def noise(mesh, magnitude=...):  # -> Any:
    """
    Add gaussian noise to every vertex of a mesh, making
    no effort to maintain topology or sanity.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Input geometry, will not be altered
    magnitude : float
      What is the maximum distance per axis we can displace a vertex.
      If None, value defaults to (mesh.scale / 100.0)

    Returns
    ----------
    permutated : trimesh.Trimesh
      Input mesh with noise applied
    """
    ...

def tessellation(mesh):  # -> Any:
    """
    Subdivide each face of a mesh into three faces with the new vertex
    randomly placed inside the old face.

    This produces a mesh with exactly the same surface area and volume
    but with different tessellation.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Input geometry

    Returns
    ----------
    permutated : trimesh.Trimesh
      Mesh with remeshed facets
    """
    ...

class Permutator:
    def __init__(self, mesh) -> None:
        """
        A convenience object to get permutated versions of a mesh.
        """
        ...
    def transform(self, translation_scale=...):  # -> Any:
        ...
    def noise(self, magnitude=...):  # -> Any:
        ...
    def tessellation(self):  # -> Any:
        ...
