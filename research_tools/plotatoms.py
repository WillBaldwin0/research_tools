import ase.io
import numpy as np
import matplotlib.pyplot as plt
try:
    from aseMolec.anaAtoms import find_molecs
except ImportError:
    pass


def offset_breakdown(eng1_lst, eng2_lst):
    eng1 = np.array(eng1_lst)
    eng2 = np.array(eng2_lst)
    offset = np.mean(eng1) - np.mean(eng2)
    reduced = eng1 - offset
    rms_error = np.sqrt(np.mean((reduced - eng2)**2))
    return offset, rms_error


def formation_energy(configs, energy_key, e0s):
    for config in configs:
        isolated_energy = sum([e0s[symbol] for symbol in config.get_chemical_symbols()])
        config.info["formation_" + energy_key] = config.info[energy_key] - isolated_energy


def scatter(idx, vals, target):
    np.add.at(target, idx.ravel(), vals.ravel())


def get_inter_forces(configs, key="AIMS_forces", keep_ats=False):
    find_molecs(configs)
    inter_forces = []
    for cfg in configs:
        num_mols = max(cfg.arrays['molID'])+1
        forces = np.zeros((num_mols,3))
        for i in range(3):
            scatter(cfg.arrays['molID'], cfg.arrays[key][:,i], forces[:,i])
        inter_forces.append(forces.flatten())
    if keep_ats:
        return inter_forces
    else:
        return np.concatenate(inter_forces)


def correlation_plot(
        ax, 
        lst1, 
        lst2=None, 
        key1=None, 
        key2=None, 
        fun1=None, 
        fun2=None, 
        in_dict="info",
        peratom=False,
        show_offset=False,
        sample=None,
        show_y_x=True,
        **kwargs,
    ):
    if lst2 is not None and len(lst2) != len(lst1):
        raise ValueError("lst1 and lst2 must have the same length")
    if len(lst1) == 0:
        print("Empty list provided, nothing to plot.")
        return

    assert (key1 is None) != (fun1 is None)
    assert (key2 is None) != (fun2 is None)
    if lst2 is None:
        lst2 = lst1

    if key1 is not None:
        if peratom:
            values1 = [getattr(item, in_dict)[key1] / len(item) for item in lst1]
        else:
            values1 = [getattr(item, in_dict)[key1] for item in lst1]
    else:
        values1 = [fun1(item) for item in lst1]
    
    if key2 is not None:
        if peratom:
            values2 = [getattr(item, in_dict)[key2] / len(item) for item in lst2]
        else:
            values2 = [getattr(item, in_dict)[key2] for item in lst2]
    else:
        values2 = [fun2(item) for item in lst2]

    # flatten them:
    values1 = np.concatenate([item.flatten() for item in values1])
    values2 = np.concatenate([item.flatten() for item in values2])

    if sample:
        values1 = values1[::len(values1)//sample]
        values2 = values2[::len(values2)//sample]    
    
    ax.scatter(values1, values2, **kwargs)
    ax.legend()
    if show_y_x:
        ax.plot([min(values1), max(values1)], [min(values1), max(values1)], color='black', alpha=0.2)
    if show_offset:
        offset, error = offset_breakdown(values1, values2)
        title = kwargs.get("title", "")
        title += f" offset={offset:.6f}, spread={error:.6f}"
        ax.set_title(title)
    return values1, values2



def dipole_corr_slab(*args, **kwargs):
    thekey1 = "" + kwargs["key1"]
    thekey2 = "" + kwargs["key2"]
    if "peratom" in kwargs and kwargs["peratom"]:
        kwargs["fun1"] = lambda x: x.info[thekey1][2] / len(x)
        kwargs["fun2"] = lambda x: x.info[thekey2][2] / len(x)
    else:
        kwargs["fun1"] = lambda x: x.info[thekey1][2]
        kwargs["fun2"] = lambda x: x.info[thekey2][2]
    del kwargs["key1"]
    del kwargs["key2"]
    kwargs["in_dict"] = "info",
    return correlation_plot(
        *args,
        **kwargs
    )