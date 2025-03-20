import numpy as np
from collections import defaultdict

def compute_lj_force(r, sigma, epsilon, rcutoff):
    """Compute Lennard-Jones force magnitude with cutoff."""
    if r >= rcutoff or r < 1e-12:
        return 0.0
    inv_r = sigma / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    return 24 * epsilon * (2 * inv_r12 - inv_r6) / r

def compute_lj_potential(r, sigma, epsilon, rcutoff):
    """Compute shifted Lennard-Jones potential with zero at cutoff."""
    if r >= rcutoff:
        return 0.0
    inv_r = sigma / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    potential = 4 * epsilon * (inv_r12 - inv_r6)

    # Shift potential to zero at cutoff
    inv_rcut = sigma / rcutoff
    inv_rcut6 = inv_rcut ** 6
    inv_rcut12 = inv_rcut6 ** 2
    shift = 4 * epsilon * (inv_rcut12 - inv_rcut6)
    return potential - shift

def compute_forces_naive(positions, box_size, rcutoff, sigma, epsilon, use_pbc=True):
    """Naive O(N^2) force calculation."""
    n = len(positions)
    forces = np.zeros_like(positions)
    potential_energy = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            if use_pbc:
                rij -= box_size * np.round(rij / box_size)
            r = np.linalg.norm(rij)
            if r < rcutoff:
                fmag = compute_lj_force(r, sigma, epsilon, rcutoff)
                forces[i] += fmag * rij / r
                forces[j] -= fmag * rij / r
                potential_energy += compute_lj_potential(r, sigma, epsilon, rcutoff)
    return forces, potential_energy

def build_linked_cells(positions, box_size, rcutoff):
    """Assign particles to cells for linked-cell algorithm (supports 2D or 3D)."""
    dim = positions.shape[1]
    n_cells = int(np.floor(box_size / rcutoff))
    n_cells = max(1, n_cells)  # Prevent 0 division
    cell_size = box_size / n_cells

    cells = defaultdict(list)
    particle_cells = []

    for idx, pos in enumerate(positions):
        cell_idx = tuple((np.floor(pos / cell_size).astype(int)) % n_cells)
        cells[cell_idx].append(idx)
        particle_cells.append(cell_idx)

    return cells, n_cells, cell_size

def compute_forces_lca(positions, box_size, rcutoff, sigma, epsilon, use_pbc=True):
    """Optimized Linked Cell Algorithm for force computation in 2D or 3D."""
    n = len(positions)
    dim = positions.shape[1]
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    rcut2 = rcutoff ** 2

    cells, n_cells, cell_size = build_linked_cells(positions, box_size, rcutoff)

    # Generate neighbor cell offsets (including self)
    shifts = [-1, 0, 1]
    neighbor_offsets = [np.array(shift) for shift in np.array(np.meshgrid(*([shifts] * dim))).T.reshape(-1, dim)]

    for cell_idx, particle_indices in cells.items():
        for i in particle_indices:
            for offset in neighbor_offsets:
                neighbor_idx = tuple((np.array(cell_idx) + offset) % n_cells)
                for j in cells.get(neighbor_idx, []):
                    if j > i:
                        rij = positions[i] - positions[j]
                        if use_pbc:
                            rij -= box_size * np.round(rij / box_size)
                        r2 = np.dot(rij, rij)
                        if r2 < rcut2 and r2 > 1e-12:
                            r = np.sqrt(r2)
                            fmag = compute_lj_force(r, sigma, epsilon, rcutoff)
                            fij = fmag * rij / r
                            forces[i] += fij
                            forces[j] -= fij
                            potential_energy += compute_lj_potential(r, sigma, epsilon, rcutoff)
    return forces, potential_energy