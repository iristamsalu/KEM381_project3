# KEM381
## Project Assignment 3: Simulation of Lennard-Jones Particles in 2D/3D Systems

### Overview
This project focuses on simulating the behavior of particles interacting via the Lennard-Jones potential. It is implemented as a molecular dynamics simulation where the primary goal is to study how particles evolve over time under different boundary conditions, interaction potentials, and computational algorithms. The project offers two main functionalities:

1. It simulates particles under the influence of the **Lennard-Jones Potential** using **Velocity Verlet** algorithm. The simulation can include **Periodic Boundary Conditions (PBC)** or **Hard-Wall** boundary conditions.

2. **Energy Minimization** involves finding the configuration of particles that minimizes the potential energy of the system by optimizing particle positions.

Additionally, the project introduces **Linked-Cell Algorithm (LCA)**, to improve performance when computing the particle interactions in large systems.

The user can run simulations in both **2D and 3D** systems, with the option to visualize particle movements and track energy.

### Files
- **main.py**: Main execution script that parses arguments, initializes the simulation, chooses between naive or linked-cell algorithms, and runs either energy minimization or full Lennard-Jones simulation with energy plotting.
- **simulation.py**: Core of the simulation logic.
Handles initialization, Velocity Verlet integration, energy minimization, boundary conditions, force calculations (LCA or naive), lattice setup, and velocity initialization.
Provides simulate_LJ() for full dynamics and minimize_energy() for energy minimization.
- **forces.py**: Implements Lennard-Jones force and potential calculations using both the naive pairwise method and the optimized LCA for efficient neighbor searching.
- **config.py**: Parses and validates command-line arguments, and stores simulation parameters in a structured Configuration dataclass.
- **plotting.py**: Handles visualization and trajectory saving.
- **requirements.txt**: A list of packages and libraries needed to run the programs.

### Installing Requirements
To install the necessary requirements in a virtual environment, use the following command:
pip3 install -r requirements.txt

### Running the Program
To run the simulation program, you need to provide certain parameters through the command line.

#### Required Arguments:
At a minimum, provide the following arguments to run the program:
python main.py --dimensions <2 or 3> --steps <number_of_steps> --dt <time_step> --density <density> --n_particles <number_of_particles>


#### Optional Arguments:
You can include the following optional arguments to further customize the simulation:
python <file_name> --dimensions <2 or 3> --steps <number_of_steps> --dt <time_step> --density <density> --n_particles <number_of_particles> --use_pbc --temperature <temperature_in_K> --sigma <LJ_sigma> --epsilon <LJ_epsilon> --rcutoff <LJ_cutoff_radius> --minimize_only --use_lca


#### Example Commands:

- **Example 1: With Periodic Boundary Conditions (PBC) in 2D**:
    ```
    python3 main.py --dimensions 2 --steps 10000 --dt 0.0001 --density 0.8 --n_particles 20 --use_pbc
    ```

    This runs the 2D simulation with **periodic boundary conditions**.

- **Example 2: With Hard Wall Boundary Conditions** in 2D:
    ```
    python3 LJ_2D.py --dimensions 2 --steps 10000 --dt 0.0001 --density 0.8 --n_particles 20
    ```

    This runs the 2D simulation with **hard wall boundary conditions** (default behavior if `--use_pbc` is not specified).

- **Example 3: With All Arguments**:
    ```
    python3 main.py --dimensions 3 --steps 10000 --dt 0.0001 --density 0.8 --n_particles 20 --use_pbc --temperature 298 --sigma 1.0 --epsilon 1.0 --rcutoff 2.5
    ```

    This runs the simulation with **periodic boundary conditions**, and specifies additional Lennard-Jones parameters, temperature, and cutoff radius.

- **Example 4: Minimize Energy**:
    ```
    python3 main.py --dimensions 2 --steps 10000 --dt 0.0001 --density 0.8 --n_particles 20 --minimize_only
    ```

    This runs the **energy minimization** with the naive algorithm in 2D, and specifies nr of steps, step length, density, and numer of particles.

- **Example 5: Minimize Energy with LCA**:
    ```
    python3 main.py --dimensions 3 --steps 10000 --dt 0.0001 --density 0.8 --n_particles 20 --minimize_only --use_lca
    ```

    This runs the **energy minimization** with **LCA** in 3D, and specifies nr of steps, step length, density, and numer of particles.

---

### Explanation of Arguments:

- `--dimensions <2 or 3>`: Set simulation to 2D or 3D
- `--steps <number_of_steps>`: The number of steps to run the simulation.
- `--dt <time_step>`: The time step used in the simulation.
- `--density <density>`: The particle density in the system.
- `--n_particles <number_of_particles>`: The number of particles in the simulation.
- `--use_pbc`: (Optional) Flag to enable **Periodic Boundary Conditions**. If omitted, **hard wall** boundary conditions are used by default.
- `--temperature <temperature_in_K>`: (Optional) The temperature in Kelvin. Used with periodic boundary conditions (`--use_pbc`).
- `--sigma <LJ_sigma>`: (Optional) The Lennard-Jones sigma parameter (distance where the potential is zero).
- `--epsilon <LJ_epsilon>`: (Optional) The Lennard-Jones epsilon parameter (depth of the potential well).
- `--rcutoff <LJ_cutoff_radius>`: (Optional) The cutoff radius for the Lennard-Jones potential.
- `--minimize_only`: (Optional) Flag to run **energy minimization**. If omitted, the regular **Lennard-Jones simulation** will be run by default.
- `--use_lca`: (Optional) Flag to run the LJ simulation or minimization using the **linked cell algorithm**. If omitted, the regular naive algorithm will be used by default.

---

### Notes:
1. The `--use_pbc` flag enables **periodic boundary conditions**, causing particles that move out of the simulation box to reappear on the opposite side. If this flag is not provided, **hard wall boundary conditions** will be used by default (particles reflect off the walls of the box).
2. Including `--temperature`, `--sigma`, `--epsilon`, and `--rcutoff` arguments will override default values and influence the simulation's behavior.
3. The `--minimize_only` flag runs only **energy minimization**. If this flag is not provided, just a regular **Lennard-Jones simulation** will be run by default.
4. The `--use_lca` flag updates forces by using **Linked Cell Algorithm**. If this flag is not provided, a naive pairwise method will be used by default.
5. Trajectories are saved to an `.xyz` file which can be visualized using tools like VMD or Ovito.
6. The energy evaluation plots are saved as `.png` files.

---

