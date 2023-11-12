"""
This type stub file was generated by pyright.
"""

from absl.testing import parameterized

"""tests for rollout function."""
TEST_XML = ...
TEST_XML_NO_SENSORS = ...
TEST_XML_NO_ACTUATORS = ...
TEST_XML_MOCAP = ...
TEST_XML_EMPTY = ...
ALL_MODELS = ...

class MuJoCoRolloutTest(parameterized.TestCase):
    def setUp(self):  # -> None:
        ...
    @parameterized.parameters(ALL_MODELS.keys())
    def test_single_step(self, model_name):  # -> None:
        ...
    @parameterized.parameters(ALL_MODELS.keys())
    def test_single_rollout(self, model_name):  # -> None:
        ...
    @parameterized.parameters(ALL_MODELS.keys())
    def test_multi_step(self, model_name):  # -> None:
        ...
    @parameterized.parameters(ALL_MODELS.keys())
    def test_single_rollout_fixed_ctrl(self, model_name):  # -> None:
        ...
    @parameterized.parameters(ALL_MODELS.keys())
    def test_multi_rollout(self, model_name):  # -> None:
        ...
    @parameterized.parameters(ALL_MODELS.keys())
    def test_multi_rollout_fixed_ctrl_infer_from_output(self, model_name):  # -> None:
        ...
    @parameterized.product(arg_nstep=[[3, 1, 1], [3, 3, 1], [3, 1, 3]], model_name=list(ALL_MODELS.keys()))
    def test_multi_rollout_multiple_inputs(self, arg_nstep, model_name):  # -> None:
        ...
    def test_threading(self):  # -> None:
        ...
    def test_time(self):  # -> None:
        ...
    def test_warmstart(self):  # -> None:
        ...
    def test_mocap(self):  # -> None:
        ...
    def test_intercept_mj_errors(self):  # -> None:
        ...
    def test_invalid(self):  # -> None:
        ...
    def test_bad_sizes(self):  # -> None:
        ...
    def test_stateless(self):  # -> None:
        ...

def get_state(data):  # -> NDArray[Unknown]:
    ...

def set_state(model, data, state):  # -> None:
    ...

def step(model, data, state, **kwargs):  # -> tuple[NDArray[Unknown], Unknown]:
    ...

def single_rollout(model, data, initial_state, **kwargs):  # -> tuple[NDArray[float64], NDArray[float64]]:
    ...

def multi_rollout(
    model, data, initial_state, **kwargs
):  # -> tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
    ...

if __name__ == "__main__": ...
