import jax.numpy as jnp
import diffrax
import equinox as eqx
from jaxtyping import Array, Float64, Int
import optimistix as optx
import matplotlib.pyplot as plt


class EquationParameters(eqx.Module):
    n: Int
    s: Float64[Array, "n"]  # noqa: F821
    end_point: Float64[Array, "2"]
    length: Float64
    mass: Float64
    wind_pressure: Float64

    @property
    def g(self) -> Float64:
        return 9.81

    def __init__(self, n, end_point, length, mass, wind_pressure):
        self.n = n
        self.s = jnp.linspace(0.0, length, num=n)
        self.end_point = end_point
        self.length = length
        self.mass = mass
        self.wind_pressure = wind_pressure


class InitialCondition(eqx.Module):
    """Initial conditions for solving the system of equations. Values at s=0"""

    x: Float64
    y: Float64
    Tx: Float64
    Ty: Float64


def system(_s, S, params: EquationParameters):
    _x, _y, Tx, Ty = S

    T = jnp.sqrt(Tx**2 + Ty**2)
    cos = Tx / T
    sin = Ty / T

    return jnp.array([cos, sin, params.mass * params.g, params.wind_pressure * cos])


def solve_ivp(ic: InitialCondition, params: EquationParameters):
    term = diffrax.ODETerm(system)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(
        rtol=1e-6,
        atol=1e-6,
    )
    y_init = jnp.array([ic.x, ic.y, ic.Tx, ic.Ty])
    saveat = diffrax.SaveAt(ts=params.s)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=params.s[0],
        t1=params.s[-1],
        dt0=0.1,
        y0=y_init,
        args=params,
        stepsize_controller=controller,
        saveat=saveat,
        adjoint=diffrax.DirectAdjoint(),
    )
    return sol


def shooting_objective(tension: Float64[Array, "2"], params: EquationParameters):
    ic = InitialCondition(0.0, 0.0, tension[0], tension[1])
    sol = solve_ivp(ic, params)
    assert sol.ys is not None, "Solver did not return any values"
    sol_end = sol.ys[-1, [0, 1]]
    return sol_end - params.end_point


def solve(tension_guess: Float64[Array, "2"], params: EquationParameters):
    solver = optx.IndirectLevenbergMarquardt(
        rtol=1e-5,
        atol=1e-5,
        verbose=frozenset({"step", "accepted", "loss", "step_size"}),
    )

    sol_root = optx.root_find(
        shooting_objective,
        solver,
        tension_guess,
        args=params,
        max_steps=1000,
        throw=False,
    )

    best_tension = sol_root.value
    ic = InitialCondition(0.0, 0.0, best_tension[0], best_tension[1])
    final_sol = solve_ivp(ic, params)
    return final_sol
