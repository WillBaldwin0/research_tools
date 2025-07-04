import ase.io
import matplotlib.pyplot as plt
import numpy as np


def plot_integrated_energy(ats_list, energy_key, force_key, name):
    energies_actual = np.array([item.info[energy_key] for item in ats_list])
    forces_actual = [item.arrays[force_key] for item in ats_list]
    positions = [item.get_positions() for item in ats_list]

    delta_positions = np.array(positions[1:]) - np.array(positions[:-1])
    midpoint_forces = (np.array(forces_actual[1:]) + np.array(forces_actual[:-1])) / 2

    work = np.sum(delta_positions * midpoint_forces, axis=(1,2))
    work = np.concatenate(([0], work))

    integrated_energy = - np.cumsum(work)

    fig, axs = plt.subplots(2, 1, figsize=(10, 16))
    axs[0].plot(integrated_energy, label='Integrated Energy')
    axs[0].plot(energies_actual - energies_actual[0], label='Actual Energy', linestyle='--')
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('Energy (eV)')
    axs[0].legend()
    axs[0].set_title('Integrated Energy vs Actual Energy')

    delta_xs = np.linalg.norm(delta_positions, axis=(-2,-1))
    cumulative_xs = np.concatenate(([0], np.cumsum(delta_xs)))
    axs[1].plot(cumulative_xs, integrated_energy, label='Integrated Energy', marker='+')
    #axs[1].scatter(cumulative_xs[::2], integrated_energy[::2], color='tab:blue')
    axs[1].plot(cumulative_xs, energies_actual - energies_actual[0], label='Actual Energy', marker='+', linestyle='--')
    #axs[1].scatter(cumulative_xs[::2], energies_actual[::2] - energies_actual[0], color='tab:orange')
    axs[1].set_xlabel('distance')
    axs[1].set_ylabel('Energy (eV)')
    axs[1].legend()
    plt.savefig(f'{name}.png', dpi=300)

