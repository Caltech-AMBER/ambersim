import jax
from flax import struct


@struct.dataclass
class ControllerParams:
    """The parameters for generic controllers.

    This is left completely empty for maximum flexibility in the API. Some examples:
        - "Regular" inputs into feedback controllers (e.g., the state) belong here.
        - Non-Markovian controllers can pass histories in this params object.
        - Parameters of the controller that you may randomize/optimize go here.
    """


@struct.dataclass
class Controller:
    """The API for a generic controller.

    See the notes in TrajectoryOptimizer on the generality of this class - much of the same applies.
    """

    def compute(self, ctrl_params: ControllerParams) -> jax.Array:
        """Computes a control input.

        Args:
            ctrl_params: ControllerParams

        Returns:
            u (shape=(nu,)): The control input.
        """
        raise NotImplementedError
