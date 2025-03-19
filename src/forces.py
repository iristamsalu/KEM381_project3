import numpy as np

def compute_lj_force(r, sigma, epsilon, rcutoff):
    """Compute the Lennard-Jones force between two particles."""
    if r >= rcutoff:
        return 0  # No force beyond the cutoff
    inv_r = sigma / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    force_mag = 24 * epsilon * (2 * inv_r12 - inv_r6) / r
    return force_mag

def compute_lj_potential(r, sigma, epsilon, rcutoff):
    """Compute the Lennard-Jones potential between two particles."""
    if r >= rcutoff:
        return 0  # No potential beyond the cutoff
    inv_r = sigma / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    potential = 4 * epsilon * (inv_r12 - inv_r6)
    return potential

def compute_forces_lca(positions, box_size, rcutoff, sigma, epsilon, use_pbc):
    """Compute forces using LCA."""
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    n_particles = len(positions)

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r_ij = positions[i] - positions[j]
            if use_pbc:
                r_ij -= box_size * np.round(r_ij / box_size)
            r = np.linalg.norm(r_ij)

            if r < rcutoff:
                force_mag = compute_lj_force(r, sigma, epsilon, rcutoff)
                force_ij = force_mag * r_ij / r

                forces[i] += force_ij
                forces[j] -= force_ij

                potential_energy += compute_lj_potential(r, sigma, epsilon, rcutoff)

    return forces, potential_energy

def compute_forces_naive(positions, box_size, rcutoff, sigma, epsilon, use_pbc):
    """Compute forces using the naive method."""
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    n_particles = len(positions)

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r_ij = positions[i] - positions[j]
            if use_pbc:
                r_ij -= box_size * np.round(r_ij / box_size)
            r = np.linalg.norm(r_ij)

            if r < rcutoff:
                force_mag = compute_lj_force(r, sigma, epsilon, rcutoff)
                force_ij = force_mag * r_ij / r

                forces[i] += force_ij
                forces[j] -= force_ij

                potential_energy += compute_lj_potential(r, sigma, epsilon, rcutoff)

    return forces, potential_energy
