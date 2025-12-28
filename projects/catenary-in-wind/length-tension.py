import jax.numpy as jnp
import jax
from jaxtyping import Float
import matplotlib.pyplot as plt
from parametric_shooting import EquationParameters, solve


@jax.jit
def tension(L: Float, fw: Float) -> Float:
    params = EquationParameters(200, jnp.array([100.0, 0.0]), L, 0.004, fw)
    sol = solve(jnp.array([1.0, -1.0]), params)
    assert sol.ys is not None, "Solver did not return any values"

    T_end = jnp.sqrt(sol.ys[-1, 2] ** 2 + sol.ys[-1, 3] ** 2)
    return T_end


fw = (9**2) / 2 * 1.225 * 1.17 * (1.5 / 1000)
lengths = jnp.linspace(101, 150, 20)

tension_vmap = jax.vmap(tension, (0, None))
T = tension_vmap(lengths, fw)

plt.plot(lengths, T)
plt.xlabel("length [m]")
plt.ylabel("Tension [N]")
plt.savefig("length-tension.png")
plt.show()
