import jax
import jax.numpy as jnp
from jax import hessian, jacobian

from ambersim.trajopt.base import CostFunctionParams
from ambersim.trajopt.cost import StaticGoalQuadraticCost
from ambersim.utils.io_utils import load_mjx_model_and_data_from_file


def test_sgqc():
    """Tests that the StaticGoalQuadraticCost works correctly."""
    # loading model and cost function
    model, _ = load_mjx_model_and_data_from_file("models/barrett_hand/bh280.xml", force_float=False)
    cost_function = StaticGoalQuadraticCost(
        Q=jnp.eye(model.nq + model.nv),
        Qf=10.0 * jnp.eye(model.nq + model.nv),
        R=0.01 * jnp.eye(model.nu),
        xg=jnp.zeros(model.nq + model.nv),
    )

    # generating dummy data
    N = 10
    key = jax.random.PRNGKey(0)
    xs = jax.random.normal(key=key, shape=(N + 1, model.nq + model.nv))
    us = jax.random.normal(key=key, shape=(N, model.nu))

    # comparing cost value vs. ground truth for loop
    val_test, _ = cost_function.cost(xs, us, params=CostFunctionParams())
    val_gt = 0.0
    for i in range(N):
        xs_err = xs[i, :] - cost_function.xg
        val_gt += 0.5 * (xs_err @ cost_function.Q @ xs_err + us[i, :] @ cost_function.R @ us[i, :])
    xs_err = xs[N, :] - cost_function.xg
    val_gt += 0.5 * (xs_err @ cost_function.Qf @ xs_err)
    val_gt = jnp.squeeze(val_gt)
    assert jnp.allclose(val_test, val_gt)

    # comparing cost gradients vs. jax autodiff
    gcost_xs_test, gcost_us_test, _, _ = cost_function.grad(xs, us, params=CostFunctionParams())
    gcost_xs_gt, gcost_us_gt, _, _ = super(StaticGoalQuadraticCost, cost_function).grad(
        xs, us, params=CostFunctionParams()
    )
    assert jnp.allclose(gcost_xs_test, gcost_xs_gt)
    assert jnp.allclose(gcost_us_test, gcost_us_gt)

    # comparing cost Hessians vs. jax autodiff
    Hcost_xsxs_test, Hcost_xsus_test, _, Hcost_usus_test, _, _, _ = cost_function.hess(
        xs, us, params=CostFunctionParams()
    )
    Hcost_xsxs_gt, Hcost_xsus_gt, _, Hcost_usus_gt, _, _, _ = super(StaticGoalQuadraticCost, cost_function).hess(
        xs, us, params=CostFunctionParams()
    )
    assert jnp.allclose(Hcost_xsxs_test, Hcost_xsxs_gt)
    assert jnp.allclose(Hcost_xsus_test, Hcost_xsus_gt)
    assert jnp.allclose(Hcost_usus_test, Hcost_usus_gt)
