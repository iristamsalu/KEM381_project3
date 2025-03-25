import argparse
import sys
import os
from dataclasses import dataclass

@dataclass
class Configuration:
    dimensions: int
    steps: int
    dt: float
    density: float
    n_particles: int
    use_pbc: bool
    temperature: float
    sigma: float
    epsilon: float
    rcutoff: float
    minimize_only: bool
    minimize: bool
    minimization_steps: int
    use_lca: bool
    use_jit: bool

def validate_positive(value, name):
    """Check if a value is positive."""
    if value <= 0:
        print(f"Error: {name} must be greater than 0.")
        sys.exit(1)

def validate_dimension(value):
    """Ensure that the dimension is either 2 or 3."""
    if value not in [2, 3]:
        print(f"Error: Dimension must be either 2 or 3, but got {value}.")
        sys.exit(1)

def parse_args():
    """Parse and validate the command line arguments."""
    parser = argparse.ArgumentParser(description="Lennard-Jones simulation")
    
    # Arguments for the simulation
    parser.add_argument("--dimensions", type=int, choices=[2, 3], default=2, help="Dimension of simulation (2 or 3)")
    parser.add_argument("--steps", type=int, default=5000, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, default=0.0001, help="Timestep")
    parser.add_argument("--density", type=float, default=0.8, help="Density of particles")
    parser.add_argument("--n_particles", type=int, default=1000, help="Number of particles")
    parser.add_argument("--use_pbc", action="store_true", help="Use periodic boundary conditions (PBC)")
    parser.add_argument("--temperature", type=float, default=298.0, help="Desired temperature (K)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Lennard-Jones sigma parameter")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Lennard-Jones epsilon parameter")
    parser.add_argument("--rcutoff", type=float, default=2.5, help="Lennard-Jones cutoff radius")
    parser.add_argument("--minimize_only", action="store_true", help="Only minimize energy")
    parser.add_argument("--minimize", action="store_true", help="Perform LJ simulation and then minimize energy")
    parser.add_argument("--minimization_steps", type=int, default=10000, help="Number of minimization steps")
    parser.add_argument("--use_lca", action="store_true", help="Use Linked Cell Algorithm (LCA)")
    parser.add_argument("--use_jit", action="store_true", help="Use Just-In-Time (JIT) optimization")

    args = parser.parse_args()

    # Validate the input
    validate_positive(args.steps, "Number of steps")
    validate_positive(args.dt, "Time step (dt)")
    validate_positive(args.density, "Density")
    validate_positive(args.n_particles, "Number of particles")
    validate_positive(args.temperature, "Temperature")
    validate_positive(args.sigma, "Sigma")
    validate_positive(args.epsilon, "Epsilon")
    validate_positive(args.rcutoff, "Cutoff radius")
    validate_positive(args.minimization_steps, "Number of minimization steps")
    validate_dimension(args.dimensions)

    # Check for rcutoff and sigma consistency
    if args.rcutoff <= 2.0 * args.sigma:
        print("Warning: Cutoff radius (rcutoff) should typically be greater than 2 * sigma for good accuracy.")

    output_dir = "output"
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)  

    return Configuration(
        args.dimensions,
        args.steps,
        args.dt,
        args.density,
        args.n_particles,
        args.use_pbc,
        args.temperature,
        args.sigma,
        args.epsilon,
        args.rcutoff,
        args.minimize_only,
        args.minimize,
        args.minimization_steps,
        args.use_lca,
        args.use_jit
    )
