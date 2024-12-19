import os
import numpy as np


def get_valid_set(folder, allset):
    import os 
    contents = [item for item in os.listdir(folder) if (os.path.isfile(os.path.join(folder, item)) and item[:13]=='valid_indices')]
    if len(contents) == 0:
        raise FileNotFoundError('no valid indcies file')
    elif len(contents) > 1:
        raise ValueError('many valid indices')
    else:
        with open(os.path.join(folder, contents[0])) as f:
            numbers = [int(line) for line in f.readlines()]
    trainset = []
    validset = []
    for i in range(len(allset)):
        if i in numbers:
            validset.append(allset[i])
        else:
            trainset.append(allset[i])
    return trainset, validset


def split_list_by_fraction(lst, fraction):
    length=len(lst)
    indices_train = np.random.choice(length, round(length*fraction))
    train = []
    test = []
    for i, item in enumerate(lst):
        if i in indices_train:
            train.append(item)
        else:
            test.append(item)
    return train, test


def split_to_configtypes(lst):
    cfg_types = {}
    for att in lst:
        if not att.info['config_type'] in cfg_types.keys():
            cfg_types[att.info['config_type']] = [att]
        else:
            cfg_types[att.info['config_type']].append(att)
    return cfg_types
