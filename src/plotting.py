import matplotlib.pyplot as plt

def plot_energy(time_steps, energies, label, filename, is_multiple=False):
    """Plot energy over time."""
    plt.figure()
    if is_multiple:
        for energy, label in energies:
            plt.plot(time_steps, energy, label=label)
        plt.legend()
    else:
        plt.plot(time_steps, energies, label=label)
    plt.xlabel("Time")
    plt.ylabel(label)
    plt.xlim(0, max(time_steps))
    plt.title(f"{label} Over Time")
    plt.savefig(filename)
    plt.clf()

def save_xyz(positions, filename, step):
    """Save particle positions to a .xyz file."""
    with open(filename, "a") as f:
        f.write(f"{len(positions)}\n")
        f.write(f"Step {step}\n")
        for i, pos in enumerate(positions, start=1):
            if len(pos) == 2:
                f.write(f"X{i} {pos[0]} {pos[1]} 0.0\n")  # 2D system
            elif len(pos) == 3:
                f.write(f"X{i} {pos[0]} {pos[1]} {pos[2]}\n")  # 3D system
