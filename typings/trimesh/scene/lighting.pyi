"""
This type stub file was generated by pyright.
"""

import numpy as np

from .. import util

"""
lighting.py
--------------

Hold basic information about lights.

Forked from the light model in `pyrender`:
https://github.com/mmatl/pyrender
"""
_DEFAULT_RGBA = np.array([60, 60, 60, 255], dtype=np.uint8)

class Light(util.ABC):
    """
    Base class for all light objects.

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (4,) uint8
        RGBA value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend
        on the type of light: point and spot lights use luminous intensity
        in candela (lm/sr) while directional lights use illuminance
        in lux (lm/m2).
    radius : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights
        Must be > 0.0
        If None, the radius is assumed to be infinite.
    """

    def __init__(self, name=..., color=..., intensity=..., radius=...) -> None: ...
    @property
    def color(self):  # -> NDArray[uint8] | Any:
        ...
    @color.setter
    def color(self, value):  # -> None:
        ...
    @property
    def intensity(self):  # -> float:
        ...
    @intensity.setter
    def intensity(self, value):  # -> None:
        ...
    @property
    def radius(self):  # -> float:
        ...
    @radius.setter
    def radius(self, value):  # -> None:
        ...

class DirectionalLight(Light):
    """
    Directional lights are light sources that act as though they are
    infinitely far away and emit light in the direction of the local -z axis.
    This light type inherits the orientation of the node that it belongs to;
    position and scale are ignored except for their effect on the inherited
    node orientation. Because it is at an infinite distance, the light is
    not attenuated. Its intensity is defined in lumens per metre squared,
    or lux (lm/m2).

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (4,) unit8
        RGBA value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend
        on the type of light.
        point and spot lights use luminous intensity in candela (lm/sr),
        while directional lights use illuminance in lux (lm/m2).
    radius : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, the radius is assumed to be infinite.
    """

    def __init__(self, name=..., color=..., intensity=..., radius=...) -> None: ...

class PointLight(Light):
    """
    Point lights emit light in all directions from their position in space;
    rotation and scale are ignored except for their effect on the inherited
    node position. The brightness of the light attenuates in a physically
    correct manner as distance increases from the light's position (i.e.
    brightness goes like the inverse square of the distance). Point light
    intensity is defined in candela, which is lumens per square radian (lm/sr).

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (4,) uint8
        RGBA value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend
        on the type of light.
        point and spot lights use luminous intensity in candela (lm/sr),
        while directional lights use illuminance in lux (lm/m2).
    radius : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, the radius is assumed to be infinite.
    """

    def __init__(self, name=..., color=..., intensity=..., radius=...) -> None: ...

class SpotLight(Light):
    """
    Spot lights emit light in a cone in the direction of the local -z axis.
    The angle and falloff of the cone is defined using two numbers, the
    `innerConeAngle` and `outerConeAngle`. As with point lights, the brightness
    also attenuates in a physically correct manner as distance increases from
    the light's position (i.e. brightness goes like the inverse square of the
    distance). Spot light intensity refers to the brightness inside the
    `innerConeAngle` (and at the location of the light) and is defined in
    candela, which is lumens per square radian (lm/sr). A spot light's position
    and orientation are inherited from its node transform. Inherited scale does
    not affect cone shape, and is ignored except for its effect on position
    and orientation.

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (4,) uint8
        RGBA value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend
        on the type of light.
        point and spot lights use luminous intensity in candela (lm/sr),
        while directional lights use illuminance in lux (lm/m2).
    radius : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, the radius is assumed to be infinite.
    innerConeAngle : float
        Angle, in radians, from centre of spotlight where falloff begins.
        Must be greater than or equal to `0` and less than `outerConeAngle`.
    outerConeAngle : float
        Angle, in radians, from centre of spotlight where falloff ends.
        Must be greater than `innerConeAngle` and less than or equal to `PI / 2.0`.
    """

    def __init__(
        self, name=..., color=..., intensity=..., radius=..., innerConeAngle=..., outerConeAngle=...
    ) -> None: ...
    @property
    def innerConeAngle(self):  # -> float:
        ...
    @innerConeAngle.setter
    def innerConeAngle(self, value):  # -> None:
        ...
    @property
    def outerConeAngle(self):  # -> float:
        ...
    @outerConeAngle.setter
    def outerConeAngle(self, value):  # -> None:
        ...

def autolight(scene):  # -> tuple[list[PointLight], list[NDArray[float64]]]:
    """
    Generate a list of lights for a scene that looks decent.

    Parameters
    --------------
    scene : trimesh.Scene
      Scene with geometry

    Returns
    --------------
    lights : [Light]
      List of light objects
    transforms : (len(lights), 4, 4) float
      Transformation matrices for light positions.
    """
    ...
