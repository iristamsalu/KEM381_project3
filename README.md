# KEM381_project3
## KEM 381 – Project Assignment 3

### Files
- **LJ_2D.py**: 2D Lennard-Jones simulator.
- **requirements.txt**: A list of packages and libraries needed to run the programs.

### Installing Requirements
To install the necessary requirements in a virtual environment, use the following command:
pip3 install -r requirements.txt

### Running the Program
To run the simulation program, you need to provide certain parameters through the command line.

#### Required Arguments:
At a minimum, provide the following arguments to run the program:
python <file_name> --steps <number_of_steps> --dt <time_step> --density <density> --n_particles <number_of_particles>


#### Optional Arguments:
You can include the following optional arguments to further customize the simulation:
python <file_name> --steps <number_of_steps> --dt <time_step> --density <density> --n_particles <number_of_particles> --use_pbc --temperature <temperature_in_K> --sigma <LJ_sigma> --epsilon <LJ_epsilon> --rcutoff <LJ_cutoff_radius>


#### Example Commands:

- **Example 1: With Periodic Boundary Conditions (PBC)**:
    ```
    python3 LJ_2D.py --steps 10000 --dt 0.0001 --density 0.8 --n_particles 20 --use_pbc
    ```

    This runs the simulation with **periodic boundary conditions**.

- **Example 2: With Hard Wall Boundary Conditions**:
    ```
    python3 LJ_2D.py --steps 10000 --dt 0.0001 --density 0.8 --n_particles 20
    ```

    This runs the simulation with **hard wall boundary conditions** (default behavior if `--use_pbc` is not specified).

- **Example 3: With All Arguments**:
    ```
    python3 LJ_2D.py --steps 10000 --dt 0.0001 --density 0.8 --n_particles 20 --use_pbc --temperature 298 --sigma 1.0 --epsilon 1.0 --rcutoff 2.5
    ```

    This runs the simulation with **periodic boundary conditions**, and specifies additional Lennard-Jones parameters, temperature, and cutoff radius.

---

### Explanation of Arguments:

- `--steps <number_of_steps>`: The number of steps to run the simulation.
- `--dt <time_step>`: The time step used in the simulation.
- `--density <density>`: The particle density in the system.
- `--n_particles <number_of_particles>`: The number of particles in the simulation.
- `--use_pbc`: (Optional) Flag to enable **Periodic Boundary Conditions**. If omitted, **hard wall** boundary conditions are used by default.
- `--temperature <temperature_in_K>`: (Optional) The temperature in Kelvin. Used with periodic boundary conditions (`--use_pbc`).
- `--sigma <LJ_sigma>`: (Optional) The Lennard-Jones sigma parameter (distance where the potential is zero).
- `--epsilon <LJ_epsilon>`: (Optional) The Lennard-Jones epsilon parameter (depth of the potential well).
- `--rcutoff <LJ_cutoff_radius>`: (Optional) The cutoff radius for the Lennard-Jones potential.

---

### Notes:
1. The `--use_pbc` flag enables **periodic boundary conditions**, causing particles that move out of the simulation box to reappear on the opposite side. If this flag is not provided, **hard wall boundary conditions** will be used by default (particles reflect off the walls of the box).
2. Including `--temperature`, `--sigma`, `--epsilon`, and `--rcutoff` arguments will override default values and influence the simulation's behavior.

---

