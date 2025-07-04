import click
from .force_checks import plot_integrated_energy
from .plotatoms import correlation_plot
import ase.io
import matplotlib.pyplot as plt


@click.group()
def cli():
    """mypackage CLI tool."""
    pass


@cli.command()
@click.argument("trajname", type=str)
@click.argument("energy_key", type=str, default="MACE_energy")
@click.argument("force_key", type=str, default="MACE_forces")
@click.option("--name", type=str, default="integrated_energy_plot")
def integrate_energy(trajname, energy_key, force_key, name):
    traj = ase.io.read(trajname, index=':')
    plot_integrated_energy(traj, energy_key, force_key, name)


@cli.command()
@click.argument("trajname", type=str)
@click.argument("key1", type=str, default="AIMS_forces")
@click.argument("key2", type=str, default="MACE_forces")
@click.option("--name", type=str, default="correlation")
@click.option("--indict", type=str, default="arrays")
@click.option("--ignore_missing_data", is_flag=True, help="Ignore missing data in the trajectory.")
@click.option("--peratom", is_flag=True)
def correlation(trajname, key1, key2, name, indict, ignore_missing_data, peratom):
    traj = ase.io.read(trajname, index=':')
    fig, ax = plt.subplots(figsize=(8, 6))
    if ignore_missing_data:
        sampled_traj = [item for item in traj if key1 in getattr(item, indict) and key2 in getattr(item, indict)]
    else:
        sampled_traj = traj

    correlation_plot(ax, sampled_traj, key1=key1, key2=key2, in_dict=indict, show_offset=True, peratom=peratom)
    ax.set_xlabel(key1)
    ax.set_ylabel(key2)
    plt.savefig(name + ".png", dpi=300)



if __name__ == "__main__":
    cli()