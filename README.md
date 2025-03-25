# KEM381
## Project Assignment 3

### Overview
This project focuses on simulating the behavior of particles interacting via the Lennard-Jones potential in 2D and 3D. It is implemented as a molecular dynamics simulation where you can use different boundary conditions, computational algorithms, change system parameters. Also, computational time and energies are tracked during the simulation. The project offers two main functionalities:

1. It simulates particles under the influence of the **Lennard-Jones Potential** using **Velocity Verlet** algorithm. The simulation can include **Periodic Boundary Conditions (PBC)** or **Hard-Wall** boundary conditions.

2. **Energy Minimization** involves finding the configuration of particles that minimizes the potential energy of the system by optimizing particle positions.

Additionally, the project introduces **Linked-Cell Algorithm (LCA)**, to improve performance when computing the particle interactions in large systems. Furthermore, the program offers an option to run the program with **Just-In-Time Compilation (JIT)** for even shorter run times.

The user can run simulations in both **2D and 3D** systems.

### Files
- **main.py**: Main execution script that parses arguments, initializes the simulation, chooses between algorithms, and runs either energy minimization or full Lennard-Jones simulation.
- **simulation.py**: Core of the simulation logic. Handles initialization, Velocity Verlet integration, energy minimization, boundary conditions, force calculations, lattice setup, and velocity initialization.
Provides simulate_LJ() for full dynamics and minimize_energy() for energy minimization.
- **forces.py**: Implements Lennard-Jones force and potential calculations using both the naive pairwise method and the optimized LCA for efficient neighbor searching.
- **forces_jit.py**: Has same functions as forces.py but uses just-in-time compilation.
- **config.py**: Parses and validates command-line arguments, and stores simulation parameters in a structured Configuration dataclass.
- **plotting.py**: Handles visualization and trajectory saving.
- **requirements.txt**: A list of packages and libraries needed to run the programs.
- **forces_parallel.py***: Has same functins as forces_jit.py but with parallel computing. To use this module, replace the content of forces_jit.py with forces_parallel.py content (name has to be forces_jit.py or running) and run with the flag --use_jit. The module is not tested.

### Installing Requirements
To install the necessary requirements in a virtual environment, use the following command:
pip3 install -r requirements.txt

### Running the Program
To run the simulation program, you need to provide certain parameters through the command line.

#### Run Lennard-Jones MD simulation with hard-walled box:
python main.py --dimensions <2 or 3> --steps <number_of_steps> --dt <time_step> --density <density> --n_particles <number_of_particles>

#### Minimize energy starting from initial lattice
python main.py --dimensions <2 or 3> --dt <time_step> --density <density> --n_particles <number_of_particles> --minimize_only --minimization_steps <number_of_steps>

#### Minimize energy running LJ simulation with PBC before minimization
python main.py --dimensions <2 or 3> --steps <number_of_steps_before_minimization> --dt <time_step> --density <density> --n_particles <number_of_particles> --use_pbc --minimize --minimization_steps <number_of_minimization_steps>

You can include these optional arguments to further customize the simulation:
--temperature <initial_temperature_in_K> --sigma <LJ_sigma> --epsilon <LJ_epsilon> --rcutoff <LJ_cutoff_radius> --use_lca --use_jit


#### Example Commands:

- **Example 1: With Periodic Boundary Conditions (PBC) in 2D**:
    ```
    python3 main.py --dimensions 2 --steps 10000 --dt 0.0001 --density 0.8 --n_particles 20 --use_pbc
    ```

    This runs the 2D simulation with **periodic boundary conditions**.

- **Example 2: With All Arguments**:
    ```
    python3 main.py --dimensions 3 --steps 10000 --dt 0.0001 --density 0.8 --n_particles 20 --use_pbc --temperature 298 --sigma 1.0 --epsilon 1.0 --rcutoff 2.5 --use_lca --use_jit
    ```

    This runs the simulation with **periodic boundary conditions**, and specifies additional Lennard-Jones parameters, temperature, and cutoff radius. Also, uses **Linked-Cell Algorithm** and **Just-In-Time Compilation** for shorter computational time.

- **Example 3: Minimize Energy from the Initial Lattice**:
    ```
    python3 main.py --dimensions 2 --dt 0.0001 --density 0.8 --n_particles 20 --minimize_only --minimization_steps 10000 
    ```

    This runs the **energy minimization** with the naive algorithm in 2D starting from the initial lattice, and specifies nr of steps, step length, density, and numer of particles.

- **Example 4: Minimize Energy with LCA and JIT**:
    ```
    python3 main.py --dimensions 3 --steps 10000 --dt 0.0001 --density 0.8 --n_particles 100 --minimize --minimization_steps 10000 --use_lca --use_jit
    ```

    This runs the **energy minimization** with **LCA** in 3D, and specifies nr of steps, step length, density, and numer of particles. It starts from random particle positions after running simulation with PBC for 10000 steps.

- **Example 5: Simulate with LCA and JIT**:
    ```
    python3 main.py --dimensions 3 --steps 10000 --dt 0.0001 --density 0.8 --n_particles 20 --use_pbc --use_lca --use_jit
    ```

    This runs the **Lennard-Jones in 3D with PBC** with **LCA** and **JIT**, and specifies nr of steps, step length, density, numer of particles.

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
- `--minimize`: (Optional) Flag to run **energy minimization** from random particle positions. If omitted, the regular **Lennard-Jones simulation** will be run by default.
- `--minimize_only`: (Optional) Flag to run **energy minimization** from initial lattice. If omitted, the regular **Lennard-Jones simulation** will be run by default.
- `--minimization_steps`: (Optional) Give only when running with `--minimize` or `--minimize_only`.
- `--use_lca`: (Optional) Flag to run the LJ simulation or minimization using the **linked cell algorithm**. If omitted, the regular naive algorithm will be used by default.
- `--use_jit`: (Optional) Flag to run the LJ simulation or minimization using the **just-in-time compilation (JIT)**.

---

### Output:
1. Program creates output folder and several output files.
2. Trajectory is saved to an `.xyz` file which can be visualized using tools like VMD or Ovito.
3. The energy evaluation plot is saved as `energy_plot.png`. Detailed energy values during the simulation are stored in `energy_data.dat`.
4. Computational times are saved to `computational_times.dat`.

---
### Notes:
The `videos/` folder contains OVITO visuals (.mp4 files) for energy minimization.

---

