"""
This type stub file was generated by pyright.
"""

"""
creation.py
--------------

Create meshes from primitives, or with operations.
"""

def revolve(linestring, angle=..., sections=..., transform=..., **kwargs):  # -> Trimesh:
    """
    Revolve a 2D line string around the 2D Y axis, with a result with
    the 2D Y axis pointing along the 3D Z axis.

    This function is intended to handle the complexity of indexing
    and is intended to be used to create all radially symmetric primitives,
    eventually including cylinders, annular cylinders, capsules, cones,
    and UV spheres.

    Note that if your linestring is closed, it needs to be counterclockwise
    if you would like face winding and normals facing outwards.

    Parameters
    -------------
    linestring : (n, 2) float
      Lines in 2D which will be revolved
    angle : None or float
      Angle in radians to revolve curve by
    sections : None or int
      Number of sections result should have
      If not specified default is 32 per revolution
    transform : None or (4, 4) float
      Transform to apply to mesh after construction
    **kwargs : dict
      Passed to Trimesh constructor

    Returns
    --------------
    revolved : Trimesh
      Mesh representing revolved result
    """
    ...

def extrude_polygon(polygon, height, transform=..., **kwargs):  # -> Trimesh:
    """
    Extrude a 2D shapely polygon into a 3D mesh

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
      2D geometry to extrude
    height : float
      Distance to extrude polygon along Z
    triangle_args : str or None
      Passed to triangle
    **kwargs : dict
      Passed to `triangulate_polygon`

    Returns
    ----------
    mesh : trimesh.Trimesh
      Resulting extrusion as watertight body
    """
    ...

def sweep_polygon(polygon, path, angles=..., **kwargs):  # -> Trimesh:
    """
    Extrude a 2D shapely polygon into a 3D mesh along an
    arbitrary 3D path. Doesn't handle sharp curvature well.


    Parameters
    ----------
    polygon : shapely.geometry.Polygon
      Profile to sweep along path
    path : (n, 3) float
      A path in 3D
    angles :  (n,) float
      Optional rotation angle relative to prior vertex
      at each vertex
    **kwargs : dict
      Passed to `triangulate_polygon`.
    Returns
    -------
    mesh : trimesh.Trimesh
      Geometry of result
    """
    ...

def extrude_triangulation(vertices, faces, height, transform=..., **kwargs):  # -> Trimesh:
    """
    Extrude a 2D triangulation into a watertight mesh.

    Parameters
    ----------
    vertices : (n, 2) float
      2D vertices
    faces : (m, 3) int
      Triangle indexes of vertices
    height : float
      Distance to extrude triangulation
    **kwargs : dict
      Passed to Trimesh constructor

    Returns
    ---------
    mesh : trimesh.Trimesh
      Mesh created from extrusion
    """
    ...

def triangulate_polygon(
    polygon, triangle_args=..., engine=..., **kwargs
):  # -> tuple[NDArray[Unknown], Unknown] | tuple[Unknown, Unknown]:
    """
    Given a shapely polygon create a triangulation using a
    python interface to `triangle.c` or mapbox-earcut.
    > pip install triangle
    > pip install mapbox_earcut

    Parameters
    ---------
    polygon : Shapely.geometry.Polygon
        Polygon object to be triangulated.
    triangle_args : str or None
        Passed to triangle.triangulate i.e: 'p', 'pq30'
    engine : None or str
      Any value other than 'earcut' will use `triangle`

    Returns
    --------------
    vertices : (n, 2) float
       Points in space
    faces : (n, 3) int
       Index of vertices that make up triangles
    """
    ...

def box(extents=..., transform=..., bounds=..., **kwargs):  # -> Trimesh:
    """
    Return a cuboid.

    Parameters
    ------------
    extents : float, or (3,) float
      Edge lengths
    transform: (4, 4) float
      Transformation matrix
    bounds : None or (2, 3) float
      Corners of AABB, overrides extents and transform.
    **kwargs:
        passed to Trimesh to create box

    Returns
    ------------
    geometry : trimesh.Trimesh
      Mesh of a cuboid
    """
    ...

def icosahedron(**kwargs):  # -> Trimesh:
    """
    Create an icosahedron, one of the platonic solids which is has 20 faces.

    Parameters
    ------------
    kwargs : dict
      Passed through to `Trimesh` constructor.

    Returns
    -------------
    ico : trimesh.Trimesh
      Icosahederon centered at the origin.
    """
    ...

def icosphere(subdivisions=..., radius=..., **kwargs):  # -> Trimesh:
    """
    Create an isophere centered at the origin.

    Parameters
    ----------
    subdivisions : int
      How many times to subdivide the mesh.
      Note that the number of faces will grow as function of
      4 ** subdivisions, so you probably want to keep this under ~5
    radius : float
      Desired radius of sphere
    kwargs : dict
      Passed through to `Trimesh` constructor.

    Returns
    ---------
    ico : trimesh.Trimesh
      Meshed sphere
    """
    ...

def uv_sphere(radius=..., count=..., transform=..., **kwargs):  # -> Trimesh:
    """
    Create a UV sphere (latitude + longitude) centered at the
    origin. Roughly one order of magnitude faster than an
    icosphere but slightly uglier.

    Parameters
    ----------
    radius : float
      Radius of sphere
    count : (2,) int
      Number of latitude and longitude lines
    kwargs : dict
      Passed thgrough
    Returns
    ----------
    mesh : trimesh.Trimesh
       Mesh of UV sphere with specified parameters
    """
    ...

def capsule(height=..., radius=..., count=..., transform=...):  # -> Trimesh:
    """
    Create a mesh of a capsule, or a cylinder with hemispheric ends.

    Parameters
    ----------
    height : float
      Center to center distance of two spheres
    radius : float
      Radius of the cylinder and hemispheres
    count : (2,) int
      Number of sections on latitude and longitude

    Returns
    ----------
    capsule : trimesh.Trimesh
      Capsule geometry with:
        - cylinder axis is along Z
        - one hemisphere is centered at the origin
        - other hemisphere is centered along the Z axis at height
    """
    ...

def cone(radius, height, sections=..., transform=..., **kwargs):  # -> Trimesh:
    """
    Create a mesh of a cone along Z centered at the origin.

    Parameters
    ----------
    radius : float
      The radius of the cylinder
    height : float
      The height of the cylinder
    sections : int or None
      How many pie wedges per revolution
    transform : (4, 4) float or None
      Transform to apply after creation
    **kwargs : dict
      Passed to Trimesh constructor

    Returns
    ----------
    cone: trimesh.Trimesh
      Resulting mesh of a cone
    """
    ...

def cylinder(radius, height=..., sections=..., segment=..., transform=..., **kwargs):  # -> Trimesh:
    """
    Create a mesh of a cylinder along Z centered at the origin.

    Parameters
    ----------
    radius : float
      The radius of the cylinder
    height : float or None
      The height of the cylinder
    sections : int or None
      How many pie wedges should the cylinder have
    segment : (2, 3) float
      Endpoints of axis, overrides transform and height
    transform : (4, 4) float
      Transform to apply
    **kwargs:
        passed to Trimesh to create cylinder

    Returns
    ----------
    cylinder: trimesh.Trimesh
      Resulting mesh of a cylinder
    """
    ...

def annulus(r_min, r_max, height=..., sections=..., transform=..., segment=..., **kwargs):  # -> Trimesh:
    """
    Create a mesh of an annular cylinder along Z centered at the origin.

    Parameters
    ----------
    r_min : float
      The inner radius of the annular cylinder
    r_max : float
      The outer radius of the annular cylinder
    height : float
      The height of the annular cylinder
    sections : int or None
      How many pie wedges should the annular cylinder have
    transform : (4, 4) float or None
      Transform to apply to move result from the origin
    segment : None or (2, 3) float
      Override transform and height with a line segment
    **kwargs:
        passed to Trimesh to create annulus

    Returns
    ----------
    annulus : trimesh.Trimesh
      Mesh of annular cylinder
    """
    ...

def random_soup(face_count=...):  # -> Trimesh:
    """
    Return random triangles as a Trimesh

    Parameters
    -----------
    face_count : int
      Number of faces desired in mesh

    Returns
    -----------
    soup : trimesh.Trimesh
      Geometry with face_count random faces
    """
    ...

def axis(origin_size=..., transform=..., origin_color=..., axis_radius=..., axis_length=...):  # -> list[Unknown] | Any:
    """
    Return an XYZ axis marker as a  Trimesh, which represents position
    and orientation. If you set the origin size the other parameters
    will be set relative to it.

    Parameters
    ----------
    transform : (4, 4) float
      Transformation matrix
    origin_size : float
      Radius of sphere that represents the origin
    origin_color : (3,) float or int, uint8 or float
      Color of the origin
    axis_radius : float
      Radius of cylinder that represents x, y, z axis
    axis_length: float
      Length of cylinder that represents x, y, z axis

    Returns
    -------
    marker : trimesh.Trimesh
      Mesh geometry of axis indicators
    """
    ...

def camera_marker(camera, marker_height=..., origin_size=...):  # -> list[Unknown | list[Unknown] | Any]:
    """
    Create a visual marker for a camera object, including an axis and FOV.

    Parameters
    ---------------
    camera : trimesh.scene.Camera
      Camera object with FOV and transform defined
    marker_height : float
      How far along the camera Z should FOV indicators be
    origin_size : float
      Sphere radius of the origin (default: marker_height / 10.0)

    Returns
    ------------
    meshes : list
      Contains Trimesh and Path3D objects which can be visualized
    """
    ...

def truncated_prisms(tris, origin=..., normal=...):  # -> Trimesh:
    """
    Return a mesh consisting of multiple watertight prisms below
    a list of triangles, truncated by a specified plane.

    Parameters
    -------------
    triangles : (n, 3, 3) float
      Triangles in space
    origin : None or (3,) float
      Origin of truncation plane
    normal : None or (3,) float
      Unit normal vector of truncation plane

    Returns
    -----------
    mesh : trimesh.Trimesh
      Triangular mesh
    """
    ...

def torus(major_radius, minor_radius, major_sections=..., minor_sections=..., transform=..., **kwargs):  # -> Trimesh:
    """Create a mesh of a torus around Z centered at the origin.

    Parameters
    ------------
    major_radius: (float)
      Radius from the center of the torus to the center of the tube.
    minor_radius: (float)
      Radius of the tube.
    major_sections: int
      Number of sections around major radius result should have
      If not specified default is 32 per revolution
    minor_sections: int
      Number of sections around minor radius result should have
      If not specified default is 32 per revolution
    transform: (4, 4) float
      Transformation matrix
    **kwargs:
      passed to Trimesh to create torus

    Returns
    ------------
    geometry : trimesh.Trimesh
      Mesh of a torus
    """
    ...
