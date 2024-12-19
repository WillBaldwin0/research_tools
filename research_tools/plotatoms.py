import ase.io
import numpy as np
import matplotlib.pyplot as plt



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
        **kwargs,
    ):
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
            values2 = [getattr(item, in_dict)[key2] / factor for item in lst2]
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

    if show_offset:
        offset, error = offset_breakdown(values1, values2)
        title = kwargs.get("title", "")
        title += " offset={}, spread={}".format(offset, error)
        kwargs["title"]=title
    
    ax.scatter(values1, values2, **kwargs)
    ax.legend()
    return values1, values2



def dipole_corr_slab(*args, **kwargs):
    print(kwargs)
    thekey1 = "" + kwargs["key1"]
    thekey2 = "" + kwargs["key2"]
    kwargs["fun1"] = lambda x: x.info[thekey1][2] 
    kwargs["fun2"] = lambda x: x.info[thekey2][2] 
    del kwargs["key1"]
    del kwargs["key2"]
    kwargs["in_dict"] = "info",
    return correlation_plot(
        *args,
        **kwargs
    )