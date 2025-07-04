from research_tools import split_to_configtypes
from typing import Any, Dict, Optional, Tuple
import numpy as np
import ase.io
from prettytable import PrettyTable


def shift_and_spread(delta_es):
    offset = np.mean(delta_es)
    reduced = delta_es - offset
    rms_error = np.sqrt(np.mean(reduced**2))
    return offset, rms_error


NEW_TABLE_TYPES = [
    "DensityCoefficientsRMSE", 
    "DensityEnergyRMSE", 
    "PerAtomRMSE",
    "DipoleRMSE",
    "DensityDipoleRMSE",
    "EnergyDensityDipoleRMSE",
    "EnergyDipolePotentialsRMSE",
    "EnergyDensityDipoleRMSEShift",
]



def evaluate(
    atoms_list,
    dft_prefix="AIMS_",
    mace_prefix="MACE_"
) -> Tuple[float, Dict[str, Any]]:
    from mace.tools.utils import (
        compute_mae,
        compute_q95,
        compute_rel_mae,
        compute_rel_rmse,
        compute_rmse,
    )

    num_configs = len(atoms_list)
    total_loss = 0.0
    E_computed = False
    delta_es_list = []
    delta_es_per_atom_list = []
    delta_fs_list = []
    Fs_computed = False
    fs_list = []
    stress_computed = False
    delta_stress_list = []
    delta_stress_per_atom_list = []
    virials_computed = False
    delta_virials_list = []
    delta_virials_per_atom_list = []
    Mus_computed = False
    delta_mus_list = []
    delta_mus_per_atom_list = []
    mus_list = []
    dmas_computed = False
    delta_dmas_list = []
    dmas_list = []
    delta_esps_list = []
    esps_list = []
    delta_polarizability_list = []
    delta_polarizability_per_atom_list = []
    batch = None  # for pylint

    for config in atoms_list:
        output = {}
        output['energy'] = config.info[mace_prefix + 'energy']
        output['forces'] = config.arrays[mace_prefix + 'forces']
        output['density_coefficients'] = config.arrays[mace_prefix + 'density_coefficients']
        output['stress'] = None # config.info.get(mace_prefix + 'stress')
        output['dipole'] = config.info.get(mace_prefix + 'dipole')

        batch = {}
        batch['energy'] = config.info[dft_prefix + 'energy']
        batch['forces'] = config.arrays[dft_prefix + 'forces']
        batch['density_coefficients'] = config.arrays[dft_prefix + 'atom_multipoles'][:,:4]
        batch['stress'] = None # config.info.get(dft_prefix + 'stress')
        batch['dipole'] = config.info.get(dft_prefix + 'dipole')


        if output.get("energy") is not None and batch["energy"] is not None:
            
            E_computed = True
            delta_es_list.append(batch["energy"] - output["energy"])
            delta_es_per_atom_list.append(
                (batch["energy"] - output["energy"]) / len(config)
            )
        if output.get("forces") is not None and batch["forces"] is not None:
            Fs_computed = True
            delta_fs_list.append(batch["forces"] - output["forces"])
            fs_list.append(batch["forces"])
        if output.get("stress") is not None and batch["stress"] is not None:
            stress_computed = True
            delta_stress_list.append(batch["stress"] - output["stress"])
            delta_stress_per_atom_list.append(
                (batch["stress"] - output["stress"])
                / len(config)
            )
        if output.get("dipole") is not None and batch.get("dipole") is not None and "config_dipole_weight" in config.info:
            assert( np.all(config.info["config_dipole_weight"]) or np.all(np.logical_not(config.info["config_dipole_weight"])))
            if config.info["config_dipole_weight"][0]:
                dipole_differences = (batch["dipole"] - output["dipole"])
                num_atoms = len(config)

                delta_mus_list.append(dipole_differences)
                delta_mus_per_atom_list.append(
                    dipole_differences / num_atoms
                )
                mus_list.append(batch["dipole"])
        if (
            output.get("density_coefficients") is not None
            and batch["density_coefficients"] is not None
        ):
            dmas_computed = True
            delta_dmas_list.append(
                batch["density_coefficients"] - output["density_coefficients"]
            )
            dmas_list.append(batch["density_coefficients"])
            #break
        esps_computed=False
        polarizability_computed=False
    

    #assert 0

    Mus_computed = len(delta_mus_list) > 0
    polars_computed = len(delta_polarizability_list) > 0

    aux = {}

    if E_computed:
        delta_es = np.array(delta_es_list)
        delta_es_per_atom = np.array(delta_es_per_atom_list)
        aux["mae_e"] = compute_mae(delta_es)
        aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
        aux["rmse_e"] = compute_rmse(delta_es)
        aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
        aux["q95_e"] = compute_q95(delta_es)

        # shift and spread
        aux["rmse_e_shift_per_atom"], aux["rmse_e_spread_per_atom"] = shift_and_spread(delta_es_per_atom)
    if Fs_computed:
        delta_fs = np.concatenate(delta_fs_list, axis=0)
        fs = np.concatenate(fs_list, axis=0)
        aux["mae_f"] = compute_mae(delta_fs)
        aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
        aux["rmse_f"] = compute_rmse(delta_fs)
        aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
        aux["q95_f"] = compute_q95(delta_fs)
    if stress_computed:
        delta_stress = np.concatenate(delta_stress_list, axis=0)
        delta_stress_per_atom = np.concatenate(delta_stress_per_atom_list, axis=0)
        aux["mae_stress"] = compute_mae(delta_stress)
        aux["rmse_stress"] = compute_rmse(delta_stress)
        aux["rmse_stress_per_atom"] = compute_rmse(delta_stress_per_atom)
        aux["q95_stress"] = compute_q95(delta_stress)
    if virials_computed:
        delta_virials = np.concatenate(delta_virials_list, axis=0)
        delta_virials_per_atom = np.concatenate(delta_virials_per_atom_list, axis=0)
        aux["mae_virials"] = compute_mae(delta_virials)
        aux["rmse_virials"] = compute_rmse(delta_virials)
        aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
        aux["q95_virials"] = compute_q95(delta_virials)
    if Mus_computed:
        delta_mus = np.concatenate(delta_mus_list, axis=0)
        delta_mus_per_atom = np.concatenate(delta_mus_per_atom_list, axis=0)
        mus = np.concatenate(mus_list, axis=0)
        aux["mae_mu"] = compute_mae(delta_mus)
        aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
        aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
        aux["rmse_mu"] = compute_rmse(delta_mus)
        aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
        aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
        aux["q95_mu"] = compute_q95(delta_mus)
    if dmas_computed:
        delta_dmas = np.concatenate(delta_dmas_list, axis=0)
        dmas = np.concatenate(dmas_list, axis=0)
        aux["mae_dma"] = compute_mae(delta_dmas)
        aux["rel_mae_dma"] = compute_rel_mae(delta_dmas, dmas)
        aux["rmse_dma"] = compute_rmse(delta_dmas)
        aux["rel_rmse_dma"] = compute_rel_rmse(delta_dmas, dmas)
        aux["q95_dma"] = compute_q95(delta_dmas)
        if delta_dmas.shape[0] > 0:
            aux['rmse_charges'] = compute_rmse(delta_dmas[:,0:1])
        if delta_dmas.shape[1] > 1:
            aux['rmse_local_dipoles'] = compute_rmse(delta_dmas[:,1:4])
    if esps_computed:
        delta_esps = np.concatenate(delta_esps_list, axis=0)
        esps = np.concatenate(esps_list, axis=0)
        aux["mae_esp"] = compute_mae(delta_esps)
        aux["rel_mae_esp"] = compute_rel_mae(delta_esps, esps)
        aux["rmse_esp"] = compute_rmse(delta_esps)
        aux["rel_rmse_esp"] = compute_rel_rmse(delta_esps, esps)
        aux["q95_esp"] = compute_q95(delta_esps)
    if polarizability_computed:
        delta_polarizability = np.concatenate(delta_polarizability_list, axis=0)
        delta_polarizability_per_atom = np.concatenate(delta_polarizability_per_atom_list, axis=0)
        aux["mae_polarizability"] = compute_mae(delta_polarizability)
        aux["rmse_polarizability"] = compute_rmse(delta_polarizability)
        aux["rmse_polarizability_per_atom"] = compute_rmse(delta_polarizability_per_atom)
        aux["q95_polarizability"] = compute_q95(delta_polarizability)


    return None, aux



def create_error_table(
    labelled_configs,
    table_type: str,
    use_pol=False
) -> PrettyTable:
    assert table_type in NEW_TABLE_TYPES
    table = PrettyTable()

    if table_type == "DensityCoefficientsRMSE":
        table.field_names = [
            "config_type", 
            "RMSE DMA / e A^l", 
            "rel DMA %",
            "RMSE qs",
            "RMSE dipoles"
        ]
    elif table_type == "DensityEnergyRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE DMA / e A^l",
        ]
    elif table_type == "PerAtomRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    elif table_type == "DipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
        ]
    elif table_type == "DensityDipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
            "RMSE DMA / e A^l", 
            "rel DMA %",
        ]
    elif table_type == "EnergyDensityDipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE DMA / e A^l",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
            "polarizability / me A^2 / V",
        ]
    elif table_type == "EnergyDipolePotentialsRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
            "RMSE ESP / mV",
            "relative ESP RMSE %",
        ]
    elif table_type == "EnergyDensityDipoleRMSEShift":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE DMA / e A^l",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
            "polarizability / me A^2 / V",
            "RMSE_energy_shift",
            "RMSE_energy_spread",
        ]
    
    if use_pol:
        for item in labelled_configs:
            item.info['AIMS_dipole'] = item.info['AIMS_polarization']
    by_cfg = split_to_configtypes(labelled_configs)
    sorted_cfgs = sorted(list(by_cfg.keys()))

    for name in sorted_cfgs:
        _, metrics = evaluate(
            by_cfg[name]
        )

        # catch missing metrics
        all_metric_name = [
            "rmse_e_per_atom",
            "rmse_f",
            "rel_rmse_f",
            "rmse_dma",
            "rel_rmse_dma",
            "rmse_charges",
            "rmse_local_dipoles",
            "rmse_mu_per_atom",
            "rel_rmse_mu",
            "rmse_esp",
            "rel_rmse_esp",
            "rmse_polarizability_per_atom",
            "rmse_e_shift_per_atom", 
            "rmse_e_spread_per_atom",
        ]
        for metric_name in all_metric_name:
            if metric_name not in metrics:
                metrics[metric_name] = "not found"
                continue
            if not ("rel" in metric_name):
                metrics[metric_name] = f"{1000 * metrics[metric_name]:.2f}"
            else:
                metrics[metric_name] = f"{metrics[metric_name]:.2f}"
        
        # add new tables here...
        if table_type == "DensityCoefficientsRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_dma'],
                    metrics['rel_rmse_dma'],
                    metrics['rmse_charges'],
                    metrics['rmse_local_dipoles'],
                ]
            )
        elif table_type == "DensityEnergyRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_e_per_atom'],
                    metrics['rmse_f'],
                    metrics['rel_rmse_f'],
                    metrics['rmse_dma'],
                ]
            )
        elif table_type == "PerAtomRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_e_per_atom'],
                    metrics['rmse_f'],
                    metrics['rel_rmse_f'],
                ]
            )
        elif table_type == "DipoleRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_mu_per_atom'],
                    metrics['rel_rmse_mu'],
                ]
            )
        elif table_type == "DensityDipoleRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_mu_per_atom'],
                    metrics['rel_rmse_mu'],
                    metrics['rmse_dma'],
                    metrics['rel_rmse_dma'],
                ]
            )
        elif table_type == "EnergyDensityDipoleRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_e_per_atom'],
                    metrics['rmse_f'],
                    metrics['rel_rmse_f'],
                    metrics['rmse_dma'],
                    metrics['rmse_mu_per_atom'],
                    metrics['rel_rmse_mu'],
                    metrics['rmse_polarizability_per_atom'],
                ]
            )
        elif table_type == "EnergyDipolePotentialsRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_e_per_atom'],
                    metrics['rmse_f'],
                    metrics['rel_rmse_f'],
                    metrics['rmse_mu_per_atom'],
                    metrics['rel_rmse_mu'],
                    metrics['rmse_esp'],
                    metrics['rel_rmse_esp'],
                ]
            )
        elif table_type == "EnergyDensityDipoleRMSEShift":
            table.add_row(
                [
                    name,
                    metrics['rmse_e_per_atom'],
                    metrics['rmse_f'],
                    metrics['rel_rmse_f'],
                    metrics['rmse_dma'],
                    metrics['rmse_mu_per_atom'],
                    metrics['rel_rmse_mu'],
                    metrics['rmse_polarizability_per_atom'],
                    metrics['rmse_e_shift_per_atom'],
                    metrics['rmse_e_spread_per_atom']
                ]
            )
        # add new tables here...
    print(table)