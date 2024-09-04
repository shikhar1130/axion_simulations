# Axion Production Mechanisms through:
### 1. Random Simulations
### 2. Loop Simulations

## Simulation set-up:
Multiple simulations of cubical grids in a static universe (scale factor $a=1$) were set up with the parameters of the field $\lambda$ and $\eta$ set to 1. The value of the gauge coupling constant was set to 0 to simulate everything in the global string model. The grid size $N=n_x n_y n_z$ where $n_x, n_y$ and $n_z$ were the number of grid points in $x, y$ and $z$ direction with grid spacing $\Delta x, \Delta y$ and $\Delta z$ were defined. The whole system was then evolved in time for certain time steps $n_t$ with spacing $\Delta t$.

To evolve the field for all the grid points, first and second-order derivatives of the field were calculated with respect to time and spatial coordinates. For this, the finite differences method was used. The evolution of energy density was done in first order at $q_0$ 
where $q$ represents the dimension and $\Delta q$ represents either $\Delta t$ or spatial spacing in between the grid points. Since this is a first-order derivative, an error of $\mathcal{O}(\Delta q)$ was introduced. This error can be compensated by choosing larger grid sizes. The Euler-Lagrange equation of motion was evolved through second-order derivatives, which induce an error of order $\mathcal{O}(\Delta q^2)$.


Because of these errors, the values of spacings in time and spatial coordinates had an upper bound. The lower bound was influenced by computation time needed to evolve the system for the required number of time steps. The simulation should also be able to resolve the core of the strings $\tau \Delta x \geq 1$, which sets constraints on the grid spacing values. Here, $\tau$ represents conformal time. The Courant–Friedrichs–Lewy condition ensures the speed of spread of information in the system remains well below $c$ or 1 (under natural units) in the numerical simulation.

 Multiple grid sizes were tested such as $128^3, 256^3, 512^3$ and $1024^3$ with parameters $\Delta x = \Delta y = \Delta z = 0.25$ with $\Delta t = 0.05$. These values were chosen after running multiple tests to check if the errors were under control. 
 
 The simulation ideally should be set up between the two phase transitions. One of them, set by the energy scale $f_a$, occurs at higher energy scales, leading to the formation of strings by breaking U(1) symmetry. The second phase transition occurs at lower energy, known as the QCD phase transition, which results in the axion being given mass. The energy scales logarithmically because of its dependence on $1/r$. This results in divergent behaviour, and the comparison of energy of string core to that of a long-range field can be represented through energy per unit length ($\mu$) where $\Delta$ and $\delta$ represent the length scale and the core width, respectively. To reach the phase transition, this value should theoretically be greater than 72 which is out of the bounds of any computer at present time.
