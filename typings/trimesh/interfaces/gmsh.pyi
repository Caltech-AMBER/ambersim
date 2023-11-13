"""
This type stub file was generated by pyright.
"""

def load_gmsh(file_name, gmsh_args=...):  # -> dict[str, dict[Unknown, Unknown]] | dict[str, Unknown]:
    """
    Returns a surface mesh from CAD model in Open Cascade
    Breap (.brep), Step (.stp or .step) and Iges formats
    Or returns a surface mesh from 3D volume mesh using gmsh.

    For a list of possible options to pass to GMSH, check:
    http://gmsh.info/doc/texinfo/gmsh.html

    An easy way to install the GMSH SDK is through the `gmsh-sdk`
    package on PyPi, which downloads and sets up gmsh:
        >>> pip install gmsh-sdk

    Parameters
    --------------
    file_name : str
      Location of the file to be imported
    gmsh_args : (n, 2) list
      List of (parameter, value) pairs to be passed to
      gmsh.option.setNumber
    max_element : float or None
      Maximum length of an element in the volume mesh

    Returns
    ------------
    mesh : trimesh.Trimesh
      Surface mesh of input geometry
    """
    ...

def to_volume(mesh, file_name=..., max_element=..., mesher_id=...):  # -> bytes | None:
    """
    Convert a surface mesh to a 3D volume mesh generated by gmsh.

    An easy way to install the gmsh sdk is through the gmsh-sdk
    package on pypi, which downloads and sets up gmsh:
        pip install gmsh-sdk

    Algorithm details, although check gmsh docs for more information:
    The "Delaunay" algorithm is split into three separate steps.
    First, an initial mesh of the union of all the volumes in the model is performed,
    without inserting points in the volume. The surface mesh is then recovered using H.
    Si's boundary recovery algorithm Tetgen/BR. Then a three-dimensional version of the
    2D Delaunay algorithm described above is applied to insert points in the volume to
    respect the mesh size constraints.

    The Frontal" algorithm uses J. Schoeberl's Netgen algorithm.
    The "HXT" algorithm is a new efficient and parallel reimplementaton
    of the Delaunay algorithm.
    The "MMG3D" algorithm (experimental) allows to generate
    anisotropic tetrahedralizations


    Parameters
    --------------
    mesh : trimesh.Trimesh
      Surface mesh of input geometry
    file_name : str or None
      Location to save output, in .msh (gmsh) or .bdf (Nastran) format
    max_element : float or None
      Maximum length of an element in the volume mesh
    mesher_id : int
      3D unstructured algorithms:
      1: Delaunay, 3: Initial mesh only, 4: Frontal, 7: MMG3D, 9: R-tree, 10: HXT

    Returns
    ------------
    data : None or bytes
      MSH data, only returned if file_name is None

    """
    ...
