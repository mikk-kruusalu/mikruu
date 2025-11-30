+++
date = '2025-07-20T20:57:40+03:00'
draft = true
title = 'Catenary in Wind'
description = "The catenary problem but with a twist"
weight = 5
math = true
+++

I came across this problem at work when we were prototyping a tethered drone system to supply in principle endless power to the drone. This can be used in natural catastrophes or military areas to cover it with a communication network. The problem described here is not really necessary for the prototyping so I took it as a personal project.

I had two main questions that I was constantly asking myself:

1. What is the shape of the cable?
2. How much extra downward force is added to the drone and how does it change with wind?

# Derivation

We look at the cable that has been attached firmly in two points $(0, 0)$ an $(x_0, y_0)$. There are two external forces acting on the cable wind $F_w$ and gravity $F_g$. Below is a figure depicting the situation. On the right there is a closeup of a small segment $ds$ from the cable. Now, the forces acting on this differential segment are shown in blue.

{{< figure src="/images/cable.png" alt="Diagram of a Loose Cable" caption="Forces Acting on a Hanging Cable" >}}

We write the force balance for both $x$ and $y$-axis
$$
\begin{align*}
&x: &\quad T_x(x+dx, y+dy) - T_x(x, y) &= \rho g ds \\
&y: &\quad -T_y(x+dx, y+dy) + T_y(x, y) &= f_w dx.
\end{align*}
$$
Here we separate the forces into perpendicular components $T_x = T \cos\alpha$ and $T_y = T \sin\alpha$.
The parameters are mass density per unit length $\rho$ and wind pressure $f_w = F_w / l$, where $l$ is the total length of the rope. From the Pythagoras we have
$$
ds = dx \sqrt{ 1 + \left( \frac{dy}{dx} \right)^2 }.
$$
Notice that when we divide both equations by $dx$ we have the definition of the derivative on the left side
$$
\begin{align*}
\frac{dT_x}{dx} &= \rho g \sqrt{ 1 + \left( \frac{dy}{dx} \right)^2 } \\
\frac{dT_y}{dx} &= -f_w.
\end{align*}
$$
The second equation is easily solved and the solution is
$$
T_y = -f_w (x - \bar{x}),
$$
where $\bar{x}$ is the integration constant. The solution is a linear function that crosses zero at $\bar{x}$. Since in the rope the forces always have to be tangential to the rope, $\bar{x}$ is the minimum point on the catenary curve. From the defintions of the force components we have that $T_x = T_y/\tan\alpha = T_y / (dy/dx)$. Using the chain rule the first equation becomes
$$
\frac{dT_y}{dx} \frac{dy}{dx} - T_y \frac{d^2y}{dx^2} = \rho g \sqrt{ 1 + \left( \frac{dy}{dx} \right)^2 } \left( \frac{dy}{dx} \right)^2.
$$
Since we know $T_y$ we can replace it and have our final equation
$$
-\frac{dy}{dx} + (x - \bar{x}) \frac{d^2y}{dx^2} = \kappa \sqrt{ 1 + \left( \frac{dy}{dx} \right)^2 } \left( \frac{dy}{dx} \right)^2,
$$
where $\kappa = \rho g / f_w$ is a constant. This is a pretty complicated differential equation! And notice that we still do not know $\bar{x}$ but it has a very clear physical meaning. So in order to get a solvable system we can add one more constraint, the length of the rope is $l$
$$
l = \int_0^{x_0} ds.
$$

# Solving the system

I used Julia and DifferentialEquation.jl for solving this system. To set up the problem for the solver, we need to write it as a system of first order equations
$$
\begin{align*}
\frac{dy}{dx} &= z \\
\frac{dz}{dx} &= \frac{ \kappa }{ x - \bar{x} } z^2 \sqrt{ 1 + z^2 } + \frac{z}{x-\bar{x}}.
\end{align*}
$$
Notice the division by $(x-\bar{x})$, this approaches infinity near the turning point and caused a lot of headache for me at first. In the code I decided to add a constant $\varepsilon = 10^{-4}$ to the numerator to improve numerical stability.

I experimented with different $\bar{x}$ manually and the numerical scheme is quite fragile. In the range $\bar{x} \in (0.3 x_0, 0.45 x_0)$ it performs quite well. However, outside of this range I could not have any of the solvers converge.

Also, I found that the dedicated [boundary value solvers](https://docs.sciml.ai/DiffEqDocs/stable/solvers/bvp_solve/), such as `MIRK4()`, converge to a physically incorrect solution, the shooting method worked the best.

In order to solve the equations derived before we need to set up a root finding problem $0 = l - \int ds$ inside of which we solve the differential equation which allows us to calculate the rope length.
