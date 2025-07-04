import ase.io 
import numpy as np


def add_total_charge(configs, value=None, key='AIMS_atom_multipoles'):
    for config in configs:
        if value is None:
            Q = np.sum(config.arrays[key][:,0])
            config.info['total_charge'] = Q
        else:
            config.info['total_charge'] = value


def strip_keys(configs, keys):
    for item in configs:
        for key in keys:
            if key in item.info.keys():
                del item.info[key]
            if key in item.arrays.keys():
                del item.arrays[key]


def exchange_keys(configs, key_pairs):
    assert type(key_pairs==list) and type(key_pairs[0] == tuple)
    for config in configs:
        for (oldkey, newkey) in key_pairs:
            if oldkey in config.info.keys():
                config.info[newkey] = config.info.pop(oldkey)
            if oldkey in config.arrays.keys():
                config.arrays[newkey] = config.arrays.pop(oldkey)


def strip_all_but(configs, keys_lst):
    for config in configs:
        infokeys_delete = []
        arraykeys_delete = []
        for key in config.info.keys():
            if key not in keys_lst:
                infokeys_delete.append(key)
        for key in config.arrays.keys():
            if key not in keys_lst:
                arraykeys_delete.append(key)

        for key in infokeys_delete:
            del config.info[key]

        for key in arraykeys_delete:
            del config.arrays[key]


def fix_aims_output_keys(configs):
    for config in configs:
        for quantity in ['dipole', 'free_energy', 'energy', 'stress']:
            if quantity in config.info:
                config.info['AIMS_' + quantity] = config.info[quantity]
                del config.info[quantity]
        for quantity in ['forces']:
            if quantity in config.arrays:
                config.arrays['AIMS_' + quantity] = config.arrays[quantity]
                del config.arrays[quantity]
        
        if 'modifed_DMA_coeficients' in config.arrays:
            config.arrays['AIMS_atom_multipoles'] = config.arrays['modifed_DMA_coeficients']
            del config.arrays['modifed_DMA_coeficients']
            del config.arrays['DMA_coeficients']


def strip_all(configs):
    strip_all_but(configs, [])


def add_dipole_from_multipoles(configs, force=False, key='AIMS_atom_multipoles'):
    for item in configs:
        qs = item.arrays[key][:,0]
        ds = item.arrays[key][:,[3,1,2]]

        charge_component = np.sum(qs[:,None] * item.get_positions(), axis=0)
        dd = np.sum(ds, axis=0)
        if not force:
            assert 'AIMS_dipole' not in item.info
        item.info['AIMS_dipole'] = dd + charge_component


def set_config_dipole_weight(configs, setting):
    assert setting in ['all', 'none', 'slab_z']
    if setting == 'all':
        for config in configs:
            config.info['config_dipole_weight'] = np.array([1.,1.,1.])
    elif setting == 'none':
        for config in configs:
            config.info['config_dipole_weight'] = np.array([0.,0.,0.])
    elif setting == 'slab_z':
        for config in configs:
            config.info['config_dipole_weight'] = np.array([0.,0.,1.])


def check_key_counts(configs):
    info_keys = {}
    arrays_keys = {}
    for config in configs:
        for key in config.info.keys():
            if key in info_keys:
                info_keys[key] += 1
            else:
                info_keys[key] = 0
        for key in config.arrays.keys():
            if key in arrays_keys:
                arrays_keys[key] += 1
            else:
                arrays_keys[key] = 0
    return info_keys, arrays_keys


def print_key_counts(configs):
    info_counts, array_counts = check_key_counts(configs)
    string = "info:"
    for key, count in info_counts.items():
        string += f'\n  {key}:            {count}'
    string += "\narrays:"
    for key, count in array_counts.items():
        string += f'\n  {key}:            {count}'
    return string
