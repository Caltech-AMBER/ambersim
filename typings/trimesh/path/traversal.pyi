"""
This type stub file was generated by pyright.
"""

def vertex_graph(entities):  # -> tuple[Graph | None, NDArray[Unknown]]:
    """
    Given a set of entity objects generate a networkx.Graph
    that represents their vertex nodes.

    Parameters
    --------------
    entities : list
       Objects with 'closed' and 'nodes' attributes

    Returns
    -------------
    graph : networkx.Graph
        Graph where node indexes represent vertices
    closed : (n,) int
        Indexes of entities which are 'closed'
    """
    ...

def vertex_to_entity_path(vertex_path, graph, entities, vertices=...):  # -> list[Unknown] | NDArray[Unknown]:
    """
    Convert a path of vertex indices to a path of entity indices.

    Parameters
    ----------
    vertex_path : (n,) int
        Ordered list of vertex indices representing a path
    graph : nx.Graph
        Vertex connectivity
    entities : (m,) list
        Entity objects
    vertices :  (p, dimension) float
        Vertex points in space

    Returns
    ----------
    entity_path : (q,) int
        Entity indices which make up vertex_path
    """
    ...

def closed_paths(entities, vertices):  # -> Any:
    """
    Paths are lists of entity indices.
    We first generate vertex paths using graph cycle algorithms,
    and then convert them to entity paths.

    This will also change the ordering of entity.points in place
    so a path may be traversed without having to reverse the entity.

    Parameters
    -------------
    entities : (n,) entity objects
        Entity objects
    vertices : (m, dimension) float
        Vertex points in space

    Returns
    -------------
    entity_paths : sequence of (n,) int
        Ordered traversals of entities
    """
    ...

def discretize_path(entities, vertices, path, scale=...):  # -> NDArray[Unknown]:
    """
    Turn a list of entity indices into a path of connected points.

    Parameters
    -----------
    entities : (j,) entity objects
       Objects like 'Line', 'Arc', etc.
    vertices: (n, dimension) float
        Vertex points in space.
    path : (m,) int
        Indexes of entities
    scale : float
        Overall scale of drawing used for
        numeric tolerances in certain cases

    Returns
    -----------
    discrete : (p, dimension) float
       Connected points in space that lie on the
       path and can be connected with line segments.
    """
    ...

class PathSample:
    def __init__(self, points) -> None: ...
    def sample(self, distances): ...
    def truncate(self, distance):  # -> ndarray[Any, dtype[Unknown]]:
        """
        Return a truncated version of the path.
        Only one vertex (at the endpoint) will be added.
        """
        ...

def resample_path(points, count=..., step=..., step_round=...):  # -> NDArray[bool_]:
    """
    Given a path along (n,d) points, resample them such that the
    distance traversed along the path is constant in between each
    of the resampled points. Note that this can produce clipping at
    corners, as the original vertices are NOT guaranteed to be in the
    new, resampled path.

    ONLY ONE of count or step can be specified
    Result can be uniformly distributed (np.linspace) by specifying count
    Result can have a specific distance (np.arange) by specifying step


    Parameters
    ----------
    points:   (n, d) float
        Points in space
    count : int,
        Number of points to sample evenly (aka np.linspace)
    step : float
        Distance each step should take along the path (aka np.arange)

    Returns
    ----------
    resampled : (j,d) float
        Points on the path
    """
    ...

def split(path):  # -> NDArray[Unknown]:
    """
    Split a Path2D into multiple Path2D objects where each
    one has exactly one root curve.

    Parameters
    --------------
    path : trimesh.path.Path2D
      Input geometry

    Returns
    -------------
    split : list of trimesh.path.Path2D
      Original geometry as separate paths
    """
    ...
