import jax.numpy as jnp
import matplotlib.pyplot as plt
from parametric_shooting import *

fw = (9**2) / 2 * 1.225 * 1.17 * (1.5 / 1000)
params = EquationParameters(200, jnp.array([100.0, 0.0]), 120, 0.004, fw)
sol = solve(jnp.array([1.0, -1.0]), params)
assert sol.ys is not None, "Solver did not return any values"

x = sol.ys[:, 0]

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(x, sol.ys[:, 1])
ax[0].set_ylabel("y [m]")

ax[1].plot(x, sol.ys[:, 2], label="Tx")
ax[1].plot(x, sol.ys[:, 3], label="Ty")
ax[1].set_ylabel("Tension [N]")
ax[1].set_xlabel("x [m]")
ax[1].legend()
plt.savefig("cable_100-120.png")
plt.show()
